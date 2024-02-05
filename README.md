# DEEP SORT
## Installation
First, clone the repository:
```
git clone https://github.com/nwojke/deep_sort.git
```
Secondly, change code in deepsort:

- open file `/deep_sort/deep_sort/linear_assignment.py.`

- Replace 
`from sklearn.utils.linear_assignment_ import linear_assignment in line 4 with from scipy.optimize import linear_sum_assignment.`

- Replace `indices = linear_assignment(cost_matrix)` in line 58 with the following lines of code:
`indices = linear_sum_assignment(cost_matrix)`

Thirdly, change:

- open file `\deep_sort\tools\generate_detections.py`

- Change code `net/%s:0` in line 82, 84 to `%s:0`
    ```
    self.input_var = tf.get_default_graph().get_tensor_by_name(
    "%s:0" % input_name)
    self.output_var = tf.get_default_graph().get_tensor_by_name(
    "%s:0" % output_name)
    ```
