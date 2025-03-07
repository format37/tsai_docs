## On this page

  * Time series classification (using raw data)
    * Data preparation:
    * Types of architectures:
  * Time series classification (using time series images)
  * Time series regression
  * Visualization



  * __Report an issue



# Tutorial notebooks

A number of tutorials have been created to help you get started to use `tsai` with time series data. Please, feel free to open the notebooks (you can open them in Colab if you want) and tweak them to do your own experiments.

## Time series classification (using raw data)

I’d recommend you to start with:

  * **Introduction to Time Series Classification**. This notebook contains a detailed walk through the steps to perform time series classification.



### Data preparation:

If you need help preparing your data you may find the following tutorials useful:

  * **Time Series data preparation** : this will show how you can do classify both univariate or multivariate time series.

  * **How to work with (very) large numpy arrays in tsai?**

  * **How to use numpy arrays in tsai?**




These last 2 provide more details in case you need them. They explain how datasets and dataloaders are created.

### Types of architectures:

Once you feel comfortable, you can start exploring different types of architectures:

  * You can use the **Time Series data preparation** notebook and replace the InceptionTime architecture by any other of your choice: 
    * MLPs
    * RNNs (LSTM, GRU)
    * CNNs (FCN, ResNet, XResNet)
    * Wavelet-based architectures
    * Transformers (like TST - 2020)
    * They all (except ROCKET) work in the same way, for univariate or multivariate time series.
  * **How to use Transformers with Time Series?** may also help you understand how to successfully apply this new type of architecture to time series.
  * You can also use **Time Series Classification Benchmark** to perform bechmarks with different architectures and/ or configurations.



ROCKET (2019) is a new technique used to generate 10-20k features from time series. These features are used in a different classifier. This is the only implementation I’m aware of that uses GPU and allows both univariate and multivariate time series. To explain this method that works very well in many cases you can use the following notebook:

  * **ROCKET: a new state-of-the-art time series classifier**



There are many types of classifiers as you can see, and it’s very difficult to know in advance which one will perform well in our task. However, the ones that have consistently deliver the best results in recent benchmark studies are **Inceptiontime** (Fawaz, 2019) and **ROCKET** (Dempster, 2019). Transformers, like **TST** (Zerveas, 2020), also show a lot of promise, but the application to time series data is so new that they have not been benchmarked against other architectures. But I’d say these are 3 architectures you should know well.

## Time series classification (using time series images)

In these tutorials, I’ve also included a section on how to transform time series into images. This will allow you to then use DL vision models like ResNet50 for example. This approach works very well in some cases, even if you have limited data. You can learn about this technique in this notebook:

  * **Imaging time series**



## Time series regression

I’ve also included an example of how you can perform time series regression with your time series using `tsai`. In this case, the label will be continuous, instead of a category. But as you will see, the use is almost identical to time series classification. You can learn more about this here:

  * **Time series regression**



## Visualization

I’ve also created PredictionDynamics callback that allows you to visualize the model’s predictions while it’s training. It can provide you some additional insights that may be useful to improve your model. Here’s the notebook:

  * PredictionDynamics



I hope you will find these tutorial useful. I’m planning to add more tutorials to demonstrate new techniques, models, etc when they become available. So stay tuned!

  * __Report an issue


