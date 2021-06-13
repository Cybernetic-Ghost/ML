import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(beauty, horror, car):
    x = np.array([beauty, horror, car])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array( [w11, w12]) # матрица 2х3
    weight2 = np.array([-1, 1]) # вектор 1х3
    
    sum_hidden = np.dot(weight1, x)
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))
    
    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))
    
    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: "+str(y))
    
    return y

beauty = 1
horror = 0
car = 1

res = go(beauty, horror, car)
if res == 1:
    print("Ты мне нравишься")
else :
    print("Созвонимся")