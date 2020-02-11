import torch
import torch.nn.functional as F
from torch.autograd import Variable

from CSA_basic_model import BasicModel
from vgg16 import VGG16FeatureExtractor


class CSAModel(BasicModel):

    def initialize(self, opt):
        BasicModel.initialize(self, opt)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.vgg = VGG16FeatureExtractor(requires_grad=False).to(self.device)

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_global.zero_()
        self.mask_global[:, :,
            int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
                self.opt.fineSize / 4) - self.opt.overlap, \
            int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
                self.opt.fineSize / 4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        



    def name(self):
        return 'CSAModel'


