from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        '''
            grad_clip: dict(max_norm=35, norm_type=2)
                            max_norm: max norm of the gradient
                            norm_type: L2
        '''
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        # operations on the gradient of parameters
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, trainer):
        # operation after call `after_train_iter`
        trainer.optimizer.zero_grad()
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()
