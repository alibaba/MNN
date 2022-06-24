def check_for_training_ops(g):
    """Check if training ops are present in the graph.

    Args:
    g: The tf.Graph on which the check for training ops needs to be
    performed.

    Raises:
        ValueError: If a training op is seen in the graph;
    """

    # The list here is obtained
    # from https://www.tensorflow.org/api_docs/cc/group/training-ops

    training_ops = frozenset([
        'ApplyAdadelta',
        'ApplyAdagrad',
        'ApplyAdagradDA',
        'ApplyAdam',
        'ApplyAddSign',
        'ApplyCenteredRMSProp',
        'ApplyFtrl',
        'ApplyFtrlV2',
        'ApplyGradientDescent',
        'ApplyMomentum',
        'ApplyPowerSign	',
        'ApplyProximalAdagrad',
        'ApplyProximalGradientDescent',
        'ApplyRMSProp',
        'ResourceApplyAdadelta',
        'ResourceApplyAdagrad',
        'ResourceApplyAdagradDA',
        'ResourceApplyAdam',
        'ResourceApplyAdamWithAmsgrad',
        'ResourceApplyAddSign',
        'ResourceApplyCenteredRMSProp',
        'ResourceApplyFtrl',
        'ResourceApplyFtrlV2',
        'ResourceApplyGradientDescent',
        'ResourceApplyKerasMomentum',
        'ResourceApplyMomentum',
        'ResourceApplyPowerSign',
        'ResourceApplyProximalAdagrad',
        'ResourceApplyProximalGradientDescent',
        'ResourceApplyRMSProp',
        'ResourceSparseApplyAdadelta',
        'ResourceSparseApplyAdagrad',
        'ResourceSparseApplyAdagradDA',
        'ResourceSparseApplyCenteredRMSProp',
        'ResourceSparseApplyFtrl',
        'ResourceSparseApplyFtrlV2',
        'ResourceSparseApplyKerasMomentum',
        'ResourceSparseApplyMomentum',
        'ResourceSparseApplyProximalAdagrad',
        'ResourceSparseApplyProximalGradientDescent',
        'ResourceSparseApplyRMSProp',
        'SparseApplyAdadelta',
        'SparseApplyAdagrad',
        'SparseApplyAdagradDA',
        'SparseApplyCenteredRMSProp',
        'SparseApplyFtrl',
        'SparseApplyFtrlV2',
        'SparseApplyMomentum',
        'SparseApplyProximalAdagrad',
        'SparseApplyProximalGradientDescent',
        'SparseApplyRMSProp',
    ])

    op_types = set([op.type for op in g.get_operations()])
    train_op_list = op_types.intersection(training_ops)

    return train_op_list

def check_for_grad_ops(g):
    grad_ops = [
        'Conv2DBackpropInput',
        'Conv2DBackpropFilter',
        'Grad', # if there is 'Grad' in op type
    ]

    allow_ops = [
        "StopGradient",
    ]

    op_types = [op.type for op in g.get_operations()]
    grad_op_list = []
    for op_type in op_types:
        if op_type in allow_ops:
            continue
        for grad_op in grad_ops:
            if grad_op in op_type:
                grad_op_list.append(op_type)

    return grad_op_list
