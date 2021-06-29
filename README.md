# getting-started-mnist

The goal of this project is to give you some basic insights regarding the platform and [neuro-flow](https://neu-ro.gitbook.io/neuro-flow/) engine workings.

Here we have shown on the [PyTorch's MNIST example](https://github.com/pytorch/examples/tree/master/mnist) how to persist datasets, launch trainings and store results on the Neu.ro platform.

Sign up at [neu.ro](https://neu.ro) and setup your local machine according to [instructions](https://docs.neu.ro/).
 
Then run:

```shell
pip install -U neuro-cli neuro-flow
neuro login
```

Start the job:
```shell
neuro-flow run [JOB_NAME]  # see the job name in .neuro/live.yml
```
e.g. start model training:
```shell
neuro-flow run train
```

Kill the job:
```shell
neuro-flow kill [JOB_NAME]  # see the job name in .neuro/live.yml
```
