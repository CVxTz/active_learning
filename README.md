# Active Learning For An Efficient Data Annotation Strategy

#### Environment :

python3.6

```
pip install -r requirements.txt
```

#### Data :

* <https://www.kaggle.com/shayanfazeli/heartbeat>
* MNIST

#### Run Experiments :

```
cd code

bash run.sh
```
Plot :
```
python plot_perf.py
```

## Project Description :

I believe that the quality and quantity of data annotations is often the most
determining factor in the success of a machine learning application. However,
manual data annotation can be very slow and costly. This is why there is a lot
of research work done on methods that reduce the need for manual annotations,
like transfer learning or unsupervised pre-training.

<br> Another way to reduce data annotation costs is to use Active Learning. It
is a set of methods that presents the data annotation steps as an interactive
process between a learning algorithm and a user, where it is the algorithm that
suggests which examples are worth annotating while the user annotates those
selected samples.

In this project, we will try to explore uncertainty based Active learning
approaches applied to MIT-BIH Arrhythmia and MNIST Datasets.<br> Since
annotations can be very pricey and sometimes require domain experts, we will
only **simulate** the process of user interaction by starting with a small
subset of the labeled dataset and subsequently only use the labels of the
samples that are suggested by the algorithm as valuable.

### Process :

![](https://cdn-images-1.medium.com/max/800/1*6XIFEisQIl3-i5KuxpUVdg.png)


#### Steps :

1 — Start with a small batch of annotated examples of start_size = 256

2 — Train classification model on the initial batch.

3 — For n steps do :<br> - Select the next batch of the most promising examples of
size batch_size using uncertainty in the form of **entropy**, where you would
rank sample by their prediction entropy and only pick the top batch_size sample
with the highest entropy.<br> - Train the model on all data selected so far.<br>
- Evaluate the model on the test set.

### Experiments :

![](https://cdn-images-1.medium.com/max/800/1*1QoiAL4GuKbFORJk5QpWvA.png)


On the MIT-BIH Arrhythmia dataset, using the full ~3000 labeled samples for
training we get : <br> - Active learning strategy : **0.80** F1 score<br> -
Random strategy : **0.74** F1 score

In order to achieve a score of **0.74** F1 score the active learning strategy
only needs ~2000 labeled examples or 2/3 of the full 3000

![](https://cdn-images-1.medium.com/max/800/1*UDpTEseHS-wh6mqPGKyvfg.png)


On the MNIST dataset, using the full ~1400 labeled samples for training we get :
<br> - Active learning strategy : **0.98** F1 score<br> - Random strategy :
**0.96** F1 score

In order to achieve a score of **0.96** F1 score the active learning strategy
only needs ~700 labeled examples or 1/2 of the full 1400

Note that the performance displayed in the plots is averaged over 10 independent
runs to reduce the effects that are due to randomness.

Also Note that I tried the exact same experiment with CIFAR10 instead of MNIST
but it did not work very well. I still don’t understand why but it is something
worth looking into in the future ☺ .

### Conclusion :

In this project we applied a simplified implementation of uncertainty based active
learning. The simulated results look encouraging in proving that active learning
can help reduce the amount of manual labels needed to achieve a good test
performance.