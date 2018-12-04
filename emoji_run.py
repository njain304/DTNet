import torch

from emoji_train import EmojiTest

digit_test = EmojiTest(use_gpu=True)

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

digit_test.init_data_loaders()
print('Data Loaders done')
digit_test.create_model()
digit_test.init_loss_function()
digit_test.init_optimizer()
print('Starting training')
kwargs = {}
kwargs["visualize_batches"] = 100
kwargs["save_batches"] = 1000
kwargs["test_batches"] = 100
digit_test.train(num_epochs=200, **kwargs)

checkpoint_name = './log/fin_model_emoji.tar'
torch.save(digit_test.log, checkpoint_name)
