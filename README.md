# DeepZAdapt

View the project write up [Learning to Hide Zonotopes Behind HyperPlanes](https://www.overleaf.com/read/wrtspffvnxzh)

## Setup instructions

You can create virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.6
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

Run the verifier from `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `fc1, fc2, fc3, fc4, fc5, conv1, conv2, conv3, conv4, conv5`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To test the verifier, you can run for example:

```bash
$ python verifier.py --net fc1 --spec ../test_cases/fc1/img0_0.06000.txt
```

To evaluate the verifier on all networks and sample test cases, we provide the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate
```
