from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ssl import *

SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_ohemce_loss': FSOhemCELoss,
    'fs_auxce_loss': FSAuxCELoss,
    'fs_aux_rmi_loss': FSAuxRMILoss,
    'fs_auxohemce_loss': FSAuxOhemCELoss,
    'segfix_loss': SegFixLoss,
    'rmi_loss': RMILoss,
    'fs_rmi_loss': FSRMILoss,
    'contrast_auxce_loss': ContrastAuxCELoss,
    'contrast_ce_loss': ContrastCELoss,
    'fs_ce_lovasz_loss': FSCELOVASZLoss,
    'ms_fs_aux_rmi_loss': MSFSAuxRMILoss,
    'fs_auxce_dsn_loss': FSAuxCELossDSN,
    'mem_contrast_ce_loss': MemContrastCELoss,
    'PCE': PCE
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if is_distributed():
            Log.info('use distributed loss')
            return loss
            
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            Log.info('use DataParallelCriterion loss')
            from lib.extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)
        Log.info('use loss: {}.'.format(key))
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)


