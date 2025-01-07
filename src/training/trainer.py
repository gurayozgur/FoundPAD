import logging
import os
from dataclasses import dataclass
from typing import List

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.evaluation import CallBackVerification
from utils.logging import (AverageMeter, CallBackLogging,
                           CallBackModelCheckpoint, CallBackTensorboard)
from utils.utils import (compute_video_score, performances_cross_db,
                         print_trainable_parameters, write_scores,
                         write_video_scores)

from .scheduler import get_scheduler


@dataclass
class TestData:
    scores: torch.Tensor
    labels: torch.Tensor
    img_pthes: List[str]
    video_ids: List[str]

########  Default Trainer ########
class Trainer():
    def __init__(self, rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header=None, test_dataloader=None, test_sampler=None):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.header = header
        self.transform = transform
        self.trainset = trainset
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.training_type = training_type
        self.config = config

        self.start_epoch = 0
        self.global_step = self.config.global_step
        self.total_step = int(len(self.trainset) / config.batch_size / self.world_size * config.num_epoch)

        # Callback
        self.callback_logging = CallBackLogging(
            config.log_every, rank, self.total_step, config.batch_size, world_size, writer=None
        )
        self.callback_verification = CallBackVerification(
            config.eval_every , rank, config.val_targets, config.eval_path, config.image_size,
            self.transform, config.batch_size_eval, config.model_name
        )
        self.callback_save_model = CallBackModelCheckpoint(rank, config.save_every, output=config.output_path)
        self.tensorboard_callback = CallBackTensorboard(rank, self.config)
        self.tensorboard_callback.log_hyperparameters()

        # Logging
        self.loss_log = AverageMeter()
        logging.info("Trainset lenght: %d" % len(self.trainset))
        logging.info("Total Step is: %d" % self.total_step)
        logging.info("Config is: {}".format(self.config.__dict__))


########################
########  CLIP  ########
########################
class TrainerClip(Trainer):
    def __init__(self, rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader=None, test_sampler=None):
        super().__init__(rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader, test_sampler)

    def start_training(self):
        if self.training_type == "text_image_contrastive":
            self.text_image_contrastive_training()
        elif self.training_type == "text_image_header":
            self.text_image_header_training()
        elif self.training_type == "image_encoder_only":
            self.image_encoder_only_training()
        elif self.training_type == "PAD_training":
            self.PAD_training()
        elif self.training_type == "PAD_training_only_header":
            self.PAD_training_only_header()
        else:
            raise ValueError()

    def text_image_contrastive_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_model,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_model,
                lr_func_drop=self.config.lr_func_drop,
                warmup_factor=1,
        )

        # Criterion
        criterion = torch.nn.CrossEntropyLoss()

        template = 'a photo of a {}.'
        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                texts =  [template.format(classname) for classname in target]
                texts = clip.tokenize(texts).to(self.rank)

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                logits_per_image, logits_per_text = self.model(images, texts)

                # loss
                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.rank)
                loss_img = criterion(logits_per_image, ground_truth)
                loss_txt = criterion(logits_per_text, ground_truth)
                total_loss = (loss_img + loss_txt) / 2
                total_loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()

                self.loss_log.update(total_loss.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=total_loss.item(),
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()

            scheduler_model.step()

            val_results = self.callback_verification(epoch, self.model)
            self.tensorboard_callback.log_verificiation(epoch, val_results)
            self.tensorboard_callback.log_on_epoch_end(epoch, self.model)
            self.callback_save_model(epoch, self.model, self.header)

        self.tensorboard_callback.close()

    # Train clip image and text encoder with same header
    def text_image_header_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_model,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_model,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header,
                lr_func_drop=self.config.lr_func_drop,
        )

        # Criterion
        criterion = torch.nn.CrossEntropyLoss()

        template = 'a photo of a {}.'
        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                texts =  [template.format(classname) for classname in target]
                texts = clip.tokenize(texts).to(self.rank)

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # text
                features_text = self.model.module.encode_text(texts)
                if self.config.loss == "AdaFace":
                    norm_text = torch.norm(features_text, 2, 1, True)
                    output_text = torch.div(features_text, norm_text)
                    thetas_text = self.header(output_text, norm_text, target)
                else:
                    thetas_text = self.header(F.normalize(features_text), target)
                loss_text = criterion(thetas_text, target)

                # image
                features_image = self.model.module.encode_image(images)
                if self.config.loss == "AdaFace":
                    norm_image = torch.norm(features_image, 2, 1, True)
                    output_image = torch.div(features_image, norm_image)
                    thetas_image = self.header(output_image, norm_image, target)
                else:
                    thetas_image = self.header(F.normalize(features_image), target)
                loss_image = criterion(thetas_image, target)

                # loss
                total_loss = (loss_image + loss_text) / 2
                total_loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(total_loss.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=total_loss.item(),
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

            val_results = self.callback_verification(epoch, self.model)
            self.tensorboard_callback.log_verificiation(epoch, val_results)
            self.tensorboard_callback.log_on_epoch_end(epoch, self.model)
            self.callback_save_model(epoch, self.model, self.header)

        self.tensorboard_callback.close()

    # Train clip image encoder only
    def image_encoder_only_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_model,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_model,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header,
                lr_func_drop=self.config.lr_func_drop,
        )

        # Criterion
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # image
                features_image = self.model.module.encode_image(images)
                if self.config.loss == "AdaFace":
                    norm_image = torch.norm(features_image, 2, 1, True)
                    output_image = torch.div(features_image, norm_image)
                    thetas_image = self.header(output_image, norm_image, target)
                else:
                    thetas_image = self.header(F.normalize(features_image), target)
                loss_image = criterion(thetas_image, target)

                loss_image.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=loss_image.item(),
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model.module.visual
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

            val_results = self.callback_verification(epoch, self.model)
            self.tensorboard_callback.log_verificiation(epoch, val_results)
            self.tensorboard_callback.log_on_epoch_end(epoch, self.model.module.visual)
            self.callback_save_model(epoch, self.model, self.header)

        self.tensorboard_callback.close()

    def gather_test_data(self, scores, labels, img_pthes, video_ids):
        local_data = TestData(
            scores=scores.cpu(), labels=labels.cpu(), img_pthes=img_pthes, video_ids=video_ids
        )
        gathered_data = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_data, local_data)

        if self.rank == 0:
            all_scores = []
            all_labels = []
            all_img_pths = []
            all_video_ids = []

            for data in gathered_data:
                all_scores.append(data.scores)
                all_labels.append(data.labels)
                all_img_pths.extend(data.img_pthes)
                all_video_ids.extend(data.video_ids)

            return (torch.cat(all_scores), torch.cat(all_labels), all_img_pths, all_video_ids)
        return None, None, None, None

    def test_model(self, epoch):
        self.model.eval()
        self.header.eval()
        results = []
        for i, testdata in enumerate(self.test_dataloader):
            raw_test_scores, gt_labels = [], []
            raw_test_video_ids = []
            raw_test_img_pths = []
            with torch.no_grad():
                for _, (raw, labels, img_pathes) in enumerate(testdata):
                    raw = raw.cuda(self.rank, non_blocking=True)
                    labels = labels.cuda(self.rank, non_blocking=True)

                    features_image = self.model.module.encode_image(raw)
                    output, _ = self.header(F.normalize(features_image), labels)
                    # print(output)
                    logits = output.softmax(dim=1)[:, 1]
                    # print(logits)
                    raw_test_scores.append(logits)
                    gt_labels.append(labels)

                    for j in range(raw.shape[0]):
                        image_name = os.path.splitext(os.path.basename(img_pathes[j]))[0]
                        video_id = os.path.join(
                            os.path.dirname(img_pathes[j]), image_name.rsplit("_", 1)[0]
                        )
                        raw_test_video_ids.append(video_id)
                        raw_test_img_pths.append(img_pathes[j])

                raw_test_scores = torch.cat(raw_test_scores)
                gt_labels = torch.cat(gt_labels)
                # print(self.rank, raw_test_scores.shape, gt_labels.shape, len(raw_test_video_ids))
                scores, labels, test_img_pathes, video_ids = self.gather_test_data(
                    raw_test_scores, gt_labels, raw_test_img_pths, raw_test_video_ids
                )
            if self.rank == 0:
                # print(self.rank, scores.shape, labels.shape, len(video_ids))
                raw_test_scores = scores.cpu().numpy()
                gt_labels = labels.cpu().numpy()
                if epoch == self.config.num_epoch - 1:
                    out_path = os.path.join(
                        self.config.output_path, self.config.test_data[i] + ".csv"
                    )
                    write_scores(test_img_pathes, raw_test_scores, gt_labels, out_path)
                raw_test_scores, gt_labels, new_video_ids = compute_video_score(
                        video_ids, raw_test_scores, gt_labels
                    )
                if epoch == self.config.num_epoch - 1:
                    video_out_path = os.path.join(
                        self.config.output_path, self.config.test_data[i] + "_video.csv"
                    )
                    write_video_scores(new_video_ids, raw_test_scores, gt_labels, video_out_path)

                raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
                raw_test_scores = (raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

                AUC_value, APCER_value, BPCER_value, HTER_value, EER_value, TH_value = performances_cross_db(raw_test_scores, gt_labels)

                results_dict = {
                    "AUC": AUC_value,
                    "APCER": APCER_value,
                    "BPCER": BPCER_value,
                    "HTER": HTER_value,
                    "EER": EER_value,
                    "TH": TH_value
                }
                results.append(results_dict)

        self.model.train()
        self.header.train()
        if self.rank == 0:
            return results
        else:
            return None

    def PAD_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_model,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_model,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header,
                lr_func_drop=self.config.lr_func_drop,
        )

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                # print(images.shape, target.shape)
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)
                # sum target and div batch size
                # print(target.sum() / target.shape[0])
                # image
                # features_image = self.model.module.encode_image(images)
                _, loss_image = self.header(F.normalize(self.model.module.encode_image(images)), target)

                loss_image.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=loss_image.item(),
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model.module.visual
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

            if epoch % 2 != 0:
                results = self.test_model(epoch)
                if self.rank == 0:
                    for i, result in enumerate(results):
                        # print the name of the test set from config.tests
                        logging.info(
                            f'Dataset: {self.config.test_data[i]}, Epoch: {epoch}, AUC: {result["AUC"]}, APCER: {result["APCER"]}, BPCER: {result["BPCER"]}, HTER: {result["HTER"]}, EER: {result["EER"]}, TH: {result["TH"]}'
                        )
                    # combined_results = {
                    #        f"{test_data[i]}_{key}": value
                    #        for i, result in enumerate(results)
                    #        for key, value in result.items()
                    #    }
                    # Log the results using the tensorboard logger
                    # self.tensorboard_callback.log_verificiation(epoch, combined_results)
                    # self.tensorboard_callback.log_on_epoch_end(epoch, self.model.module.visual)

            self.callback_save_model(epoch, self.model, self.header)
        self.tensorboard_callback.close()

    def PAD_training_only_header(self):
        # Freeze the clip model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header,
                lr_func_drop=self.config.lr_func_drop,
        )

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # image
                with torch.no_grad():
                    features_image = self.model.module.encode_image(images)
                _, loss_image = self.header(F.normalize(features_image), target)

                loss_image.backward()

                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=loss_image.item(),
                    learning_rate=scheduler_header.get_last_lr()[0],
                    model=self.model.module.visual
                )

                self.callback_logging(self.global_step, self.loss_log, epoch)
                optimizer_header.zero_grad()

            scheduler_header.step()

            if epoch % 2 != 0:
                results = self.test_model(epoch)
                if self.rank == 0:
                    for i, result in enumerate(results):
                        logging.info(
                            f'Dataset: {self.config.test_data[i]}, Epoch: {epoch}, AUC: {result["AUC"]}, APCER: {result["APCER"]}, BPCER: {result["BPCER"]}, HTER: {result["HTER"]}, EER: {result["EER"]}, TH: {result["TH"]}'
                        )
                    # combined_results = {
                    #        f"{test_data[i]}_{key}": value
                    #        for i, result in enumerate(results)
                    #        for key, value in result.items()
                    #    }
                    # Log the results using the tensorboard logger
                    # self.tensorboard_callback.log_verificiation(epoch, combined_results)
                    # self.tensorboard_callback.log_on_epoch_end(epoch, self.model.module.visual)
            self.callback_save_model(epoch, self.model, self.header)
        self.tensorboard_callback.close()
