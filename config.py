from easydict import EasyDict

cfg = EasyDict()

cfg.DATA = EasyDict()
cfg.DATA.RESOLUTION = (352, 352)
cfg.DATA.SALICON_ROOT = ".\dataset\salicon"
cfg.DATA.SALICON_TRAIN = ".\dataset\salicon_train.csv"
cfg.DATA.SALICON_VAL = ".\dataset\salicon_val.csv"
cfg.DATA.LOG_DIR = "./logs"


# Train
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 10

cfg.SOLVER = EasyDict()
cfg.SOLVER.LR = 1e-4
cfg.SOLVER.MIN_LR = 1e-8
cfg.SOLVER.MAX_EPOCH = 15