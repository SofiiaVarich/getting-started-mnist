# getting-started-mnist

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
