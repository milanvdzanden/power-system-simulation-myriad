# power-system-simulation

This is a student project for Power System Simulation.


## Installation

In a Python environment, in the root of the repository, install it in develop mode using the command below.

**NOTE: you need to re-run the following command everytime you add new (optional) dependencies!**

```shell
pip install -e .[dev,example]
```

After installation, run the test.

```shell
pytest
```

## Code style and quality check

You can run the following two commands to automatically format your code style.

```shell
isort .
black .
```

You can run the following command to check the code quality.
It will return errors if the quality check fails.
You need to read the errors and make required adjustments.

```shell
pylint power_system_simulation 
```
