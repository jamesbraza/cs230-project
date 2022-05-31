from training.rand_search_func import perform_random_search
import matplotlib.pyplot as plt


x,y, train_loss, train_accu, val_loss, val_accu = perform_random_search()


plt.yscale('log')
plt.xlim([0, 1])
plt.scatter(x,y)
plt.xlabel("dropout rate")
plt.ylabel("lambda for L2 regularization")
