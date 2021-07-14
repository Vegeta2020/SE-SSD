# from torch.nn.utils import clip_grad
#
# from .hook import Hook
#
#
# class OptimizerHook(Hook):
#     def __init__(self, grad_clip=None):
#         '''
#             grad_clip: dict(max_norm=35, norm_type=2)
#                             max_norm: max norm of the gradient
#                             norm_type: L2
#         '''
#         self.grad_clip = grad_clip
#
#     def clip_grads(self, params):
#         # operations on the gradient of parameters
#         clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **self.grad_clip)
#
#     def after_train_iter(self, trainer):
#         # operation after call `after_train_iter`
#         trainer.optimizer.zero_grad()
#         trainer.outputs["loss"].backward()
#         if self.grad_clip is not None:
#             self.clip_grads(trainer.model.parameters())
#         trainer.optimizer.step()


# 2020/10/10: my new version of optimizer, it only add the invalid_to_zero to make the nan grad be zero,
# and the nan is caused by the 3d iou loss in cyclist training. And the above commented code is the default version.
from torch.nn.utils import clip_grad

from .hook import Hook
import torch

class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        '''
            grad_clip: dict(max_norm=35, norm_type=2)
                            max_norm: max norm of the gradient
                            norm_type: L2
        '''
        self.grad_clip = grad_clip

    def invalid_to_zero(self, params):
        for p in params:
            if p.requires_grad and torch.isnan(p.grad.data).any():
                p.grad.data[torch.isnan(p.grad.data)] = 0.0
        return params

    def clip_grads(self, params):
        # operations on the gradient of parameters
        # params = self.invalid_to_zero(params)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, trainer):
        # operation after call `after_train_iter`
        trainer.optimizer.zero_grad()
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()
