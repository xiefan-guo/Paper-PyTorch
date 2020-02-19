import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from .dataset import ImageDataset
from .model import EdgeModel, InpaintingModel
from .metric import EdgeAccuracy, PSNR
from .util import Progbar, create_dir, stitch_image, imsave, imshow


class EdgeConnect():

    def __init__(self, config):

        self.config = config
        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MOEDL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpainting_model = InpaintingModel(config).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = ImageDataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = ImageDataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = ImageDataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):

        if self.config.MODEL == 1:
            self.edge_model.load()
        elif self.config.MODEL == 2:
            self.inpainting_model.load()
        else:
            self.edge_model.load()
            self.inpainting_model.load()

    def save(self):

        if self.config.MODEL == 1:
            self.edge_model.save()
        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpainting_model.save()
        else:
            self.edge_model.save()
            self.inpainting_model.save()

    def log(self, logs):
        # ----------
        # 参数'a'代表追加
        # ----------
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # ------------------
        # [0, 1] => [0, 255]
        # ------------------
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    # --------
    # training
    # --------
    def train(self):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(self.config.MAX_ITERS)
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):

            epoch += 1
            print('Training epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:

                self.edge_model.train()
                self.inpainting_model.train()

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metric
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration

                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpainting_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metric
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpainting_model.backward(gen_loss, dis_loss)
                    iteration = self.inpainting_model.iteration

                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = (outputs * masks) + (edges * (1 - masks))
                    else:
                        outputs = edges
                    outputs, gen_loss, dis_loss, logs = self.inpainting_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metric
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpainting_model.backward(gen_loss, dis_loss)
                    iteration = self.inpainting_model.iteration

                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpainting_model.process(images, e_outputs, masks)
                    outputs_merged = i_outputs * masks + images * (1 - masks)

                    # metric
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpainting_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpainting_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ('epoch', epoch),
                    ('iter', iteration)
                ] + logs

                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('End training...')

    def eval(self):

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        self.edge_model.eval()
        self.inpainting_model.eval()

        for items in val_loader:

            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metric
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))

            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpainting_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpainting_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            # joint
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpainting_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs

            logs = [
                ('it', iteration)
            ] + logs

            progbar.add(len(images), values=logs)

    def test(self):

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1
        )

        model = self.config.MODEL
        create_dir(self.results_path)

        self.edge_model.eval()
        self.inpainting_model.eval()

        index = 0
        for items in test_loader:

            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpainting_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model | joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpainting_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('End test...')

    def sample(self, it=None):

        # don't sample when validation dataset is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpainting_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpainting_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpainting_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model | joint model
        else:
            iteration = self.inpainting_model.iteration
            inputs = images * (1 - masks) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpainting_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        img_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            img_per_row = 1

        images = stitch_image(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=img_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

