# Competition Info

## Submission Guidelines

A typical submission will be a **zip** file consisting of following files:

* `requirements.txt` - Python dependencies, **the name must be exactly "requirements.txt"**!
* `submission.py` - Training script, **the name must be exactly "submission.py"**!
* `model`: model

For the `submission.py` file, the content should be like this:

```python
class CustomedAgent:
    ''' Custom agent class, note that the class name must be "CustomAgent",
    '''
    def act(self,state_infos):
        ''' note that the function name must be "act",
            and the return must be correct format.
        '''
        action_dict = {}
        return action_dict
```

In addition to the aforementioned mandatory content, you can freely define other content. We have provided two sample submission files; one is a random strategy, and the other is an RL baseline strategy, which you can refer to as needed.