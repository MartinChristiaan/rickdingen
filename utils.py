import numpy as np
import torch

def test_knapsack(x_weights, x_prices, x_capacity, picks):
    total_price = np.dot(x_prices, picks)
    total_weight = np.dot(x_weights, picks)
    return total_price, max(total_weight - x_capacity, 0)


def brute_force_knapsack(x_weights, x_prices, x_capacity):
    picks_space = 2 ** x_weights.shape[0]
    best_price = 0
    best_picks = None
    for p in range(picks_space):
        picks = np.zeros((x_weights.shape[0]))
        for i, c in enumerate("{0:b}".format(p)[::-1]):
            picks[i] = c
        price, violation = test_knapsack(x_weights, x_prices, x_capacity, picks)
        if violation == 0:
            if price > best_price:
                best_price = price
                best_picks = picks
    return best_price, best_picks

def create_knapsack(item_count=5):
    x_weights = np.random.randint(1, 15, item_count)
    x_prices = np.random.randint(1, 10, item_count)
    x_capacity = np.random.randint(15, 50)
    _, y_best_picks = brute_force_knapsack(x_weights, x_prices, x_capacity)
    return x_weights, x_prices, np.array(x_capacity), y_best_picks

def metric_overprice(input_prices):
    def overpricing(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.batch_dot(y_pred, input_prices, 1) - K.batch_dot(y_true, input_prices, 1))
    return overpricing

def metric_space_violation(input_weights, input_capacity):
    def space_violation(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.maximum(K.batch_dot(y_pred, input_weights, 1) - input_capacity, 0))

    return space_violation

def metric_pick_count():
    def pick_count(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.sum(y_pred, -1) - K.sum(y_true, -1))

    return pick_count

def create_knapsack_dataset(count, item_count=5):
    x = [[], [], []]
    y = [[]]
    for _ in range(count):
        p = create_knapsack(item_count)
        x[0].append(p[0])
        x[1].append(p[1])
        x[2].append(p[2])
        y[0].append(p[3])
    
    
    return x, y

def torch_batch_dot(A, B):
    return torch.bmm(A.view(A.shape[0], 1, A.shape[1]), B.view(A.shape[0], A.shape[1], 1))