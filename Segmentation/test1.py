from solver import Solver
from architecture.unet import Unet


model = Unet()
solver = Solver(model.build_model()
                ,[], []
                ,num_train_examples = 10
                ,num_val_examples = 2
                ,metric = 'dice_loss'
               )

print(solver._optim_config)
