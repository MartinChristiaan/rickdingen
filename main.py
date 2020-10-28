import torch
import numpy as np
import utils

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.set_grad_enabled(True)

train_x, train_y = utils.create_knapsack_dataset(1000)
test_x, test_y = utils.create_knapsack_dataset(200)
for i in range(3):
    train_x[i] = torch.tensor(np.array(train_x[i]))
    test_x[i] = torch.tensor(np.array(test_x[i]))

train_y =  torch.tensor(np.array(train_y)).cuda().float()

input_weights, input_prices, input_capacity = [inp.cuda().float() for inp in train_x]


train_x = torch.cat(train_x[:2] + [train_x[-1].unsqueeze(1)],dim=1).cuda().float()

#test_x = torch.cat(test_x[:2] + [test_x[-1].unsqueeze(1)],dim=1)

model = nn.Sequential(
          nn.Linear(11,5),
          nn.Sigmoid(),
        ).cuda()
from torch.autograd import Variable

#%%
num_epochs = 1000
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss(reduction='sum')
loss_figure = []
# cvc = Class Variance Control 
# def loss(y_true, y_pred, cvc=200):
#     value_component = utils.torch_batch_dot(y_pred,input_prices)[:,0,0]
#     weight_component = utils.torch_batch_dot(y_pred,input_weights)[:,0,0] - input_capacity
#     zeros = torch.zeros_like(weight_component)
#     weight_component= torch.where(weight_component<0,weight_component,zeros)
#     out = -1 * value_component + cvc * weight_component
    # out[out<0] = 0
#    return out.sum()




for epoch in tqdm(range(num_epochs)):    
    y_pred = model(train_x)
    gekke_loss = loss_fn(y_pred,train_y[0])
    optimizer.zero_grad()
    
    gekke_loss.backward()
    optimizer.step()
    loss_figure += [gekke_loss.item()]

plt.figure()
plt.plot(loss_figure)
plt.show





#     def supervised_continues_knapsack(item_count=5):
#         input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     picks = Dense(item_count, use_bias=False, activation="sigmoid")(inputs_concat)
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   binary_crossentropy,
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model


# def train_knapsack(model):
#     from keras.callbacks import ModelCheckpoint
#     import os
#     if os.path.exists("best_model.h5"): os.remove("best_model.h5")
#     model.fit([train_x1, train_x2, train_x3], train_y, epochs=96, callbacks=[ModelCheckpoint("best_model.h5", monitor="loss", save_best_only=True, save_weights_only=True)])
#     model.load_weights("best_model.h5")
#     train_results = model.evaluate([train_x1, train_x2, train_x3], train_y, 64, 0)
#     test_results = model.evaluate([test, test_x2, test_x3], test_y, 64, 0)
#     print("Model results(Train/Test):")
#     print(f"Loss:               {train_results[0]:.2f} / {test_results[0]:.2f}")
#     print(f"Binary accuracy:    {train_results[1]:.2f} / {test_results[1]:.2f}")
#     print(f"Space violation:    {train_results[2]:.2f} / {test_results[2]:.2f}")
#     print(f"Overpricing:        {train_results[3]:.2f} / {test_results[3]:.2f}")
#     print(f"Pick count:         {train_results[4]:.2f} / {test_results[4]:.2f}")
    

#     model = supervised_continues_knapsack()
# train_knapsack(model)

# def supervised_continues_knapsack_one_hidden(item_count=5):
#     input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     picks = Dense(item_count * 10, use_bias=False, activation="sigmoid")(inputs_concat)
#     picks = Dense(item_count, use_bias=False, activation="sigmoid")(picks)
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   binary_crossentropy,
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model
# model = supervised_continues_knapsack_one_hidden()
# train_knapsack(model)

# def supervised_discrete_knapsack(item_count=5):
#     input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     concat_tanh = Dense(item_count, use_bias=False, activation="tanh")(inputs_concat)
#     concat_sigmoid = Dense(item_count, use_bias=False, activation="sigmoid")(inputs_concat)
#     concat_multiply = Multiply()([concat_sigmoid, concat_tanh])
#     picks = Multiply()([concat_multiply, concat_multiply])
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   binary_crossentropy,
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model

# model = supervised_discrete_knapsack()
# train_knapsack(model)

# def knapsack_loss(input_weights, input_prices, input_capacity, cvc=1):
#     def loss(y_true, y_pred):
#         picks = y_pred
#         return (-1 * K.batch_dot(picks, input_prices, 1)) + cvc * K.maximum(
#             K.batch_dot(picks, input_weights, 1) - input_capacity, 0)
#     return loss

#     def unsupervised_discrete_knapsack(item_count=5):
#         input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     concat_tanh = Dense(item_count, use_bias=False, activation="tanh")(inputs_concat)
#     concat_sigmoid = Dense(item_count, use_bias=False, activation="sigmoid")(inputs_concat)
#     concat_multiply = Multiply()([concat_sigmoid, concat_tanh])
#     picks = Multiply()([concat_multiply, concat_multiply])
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   knapsack_loss(input_weights, input_prices, input_capacity, 1),
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model
# model = unsupervised_discrete_knapsack()
# train_knapsack(model)

# def unsupervised_discrete_knapsack_one_hidden(item_count=5):
#     input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     inputs_concat = Dense(item_count * 10, activation="relu")(inputs_concat)
#     concat_tanh = Dense(item_count, use_bias=False, activation="tanh")(inputs_concat)
#     concat_sigmoid = Dense(item_count, use_bias=False, activation="sigmoid")(inputs_concat)
#     concat_multiply = Multiply()([concat_sigmoid, concat_tanh])
#     picks = Multiply()([concat_multiply, concat_multiply])
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   knapsack_loss(input_weights, input_prices, input_capacity),
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model
# model = unsupervised_discrete_knapsack_one_hidden()
# train_knapsack(model)