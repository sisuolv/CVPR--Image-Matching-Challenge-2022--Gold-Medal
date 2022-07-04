from src2.config.default import _CN as cfg

cfg.LOFTR.COARSE.TEMP_BUG_FIX = False
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'sinkhorn'
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
