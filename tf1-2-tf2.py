import re
from pathlib import Path

replacements = {
    # Sessions / placeholders / initialization
    r'(?<!compat\.v1\.)tf\.Session\(':
        'tf.compat.v1.Session(',
    r'(?<!compat\.v1\.)tf\.placeholder\(':
        'tf.compat.v1.placeholder(',
    r'(?<!compat\.v1\.)tf\.global_variables_initializer\(':
        'tf.compat.v1.global_variables_initializer(',

    # Graph-level stuff
    r'(?<!compat\.v1\.)tf\.reset_default_graph\(':
        'tf.compat.v1.reset_default_graph(',
    r'(?<!compat\.v1\.)tf\.get_default_graph\(':
        'tf.compat.v1.get_default_graph(',

    # Config / sessions (test.py, train.py)
    r'(?<!compat\.v1\.)tf\.ConfigProto\(':
        'tf.compat.v1.ConfigProto(',

    # Train helpers
    r'(?<!compat\.v1\.)tf\.train\.Saver\(':
        'tf.compat.v1.train.Saver(',
    r'(?<!compat\.v1\.)tf\.train\.latest_checkpoint\(':
        'tf.compat.v1.train.latest_checkpoint(',
    r'(?<!compat\.v1\.)tf\.train\.AdamOptimizer\(':
        'tf.compat.v1.train.AdamOptimizer(',
    r'(?<!compat\.v1\.)tf\.train\.exponential_decay\(':
        'tf.compat.v1.train.exponential_decay(',
    r'(?<!compat\.v1\.)tf\.trainable_variables\(':
        'tf.compat.v1.trainable_variables(',

    # Collections / GraphKeys (train.py)
    r'(?<!compat\.v1\.)tf\.get_collection\(':
        'tf.compat.v1.get_collection(',
    r'(?<!compat\.v1\.)tf\.GraphKeys\.':
        'tf.compat.v1.GraphKeys.',

    # Variable API (UnrollNet, networks)
    r'(?<!compat\.v1\.)tf\.variable_scope\(':
        'tf.compat.v1.variable_scope(',
    r'(?<!compat\.v1\.)tf\.get_variable\(':
        'tf.compat.v1.get_variable(',
    r'(?<!compat\.v1\.)tf\.get_variable_scope\(':
        'tf.compat.v1.get_variable_scope(',

    # tf.train.import_meta_graph (test.py)
    r'(?<!compat\.v1\.)tf\.train\.import_meta_graph\(':
        'tf.compat.v1.train.import_meta_graph(',

    # Summaries
    r'(?<!compat\.v1\.)tf\.summary\.merge_all\(':
        'tf.compat.v1.summary.merge_all(',
    r'(?<!compat\.v1\.)tf\.summary\.FileWriter\(':
        'tf.compat.v1.summary.FileWriter(',

    # Layers (if any remaining)
    r'(?<!compat\.v1\.)tf\.layers\.conv2d\(':
        'tf.compat.v1.layers.conv2d(',
    r'(?<!compat\.v1\.)tf\.layers\.dense\(':
        'tf.compat.v1.layers.dense(',
    r'(?<!compat\.v1\.)tf\.layers\.batch_normalization\(':
        'tf.compat.v1.layers.batch_normalization(',

    # Logging
    r'(?<!compat\.v1\.)tf\.logging\.':
        'tf.compat.v1.logging.',

    # Random seed + random initializer (networks.py)
    r'(?<!compat\.v1\.)tf\.set_random_seed\(':
        'tf.compat.v1.set_random_seed(',
    r'(?<!compat\.v1\.)tf\.random_normal_initializer\(':
        'tf.compat.v1.random_normal_initializer(',
}

# Complex pattern: tf.cast(x, tf.float32) → tf.cast(x, tf.float32)
to_float_pattern = re.compile(r'tf\.to_float\(([^)]+)\)')


def replace_to_float(text: str) -> str:
    return to_float_pattern.sub(r'tf.cast(\1, tf.float32)', text)


# Apply to all .py files recursively
py_files = list(Path('.').rglob('*.py'))

for path in py_files:
    original = path.read_text()
    new = original

    # simple replacements
    for pattern, repl in replacements.items():
        new = re.sub(pattern, repl, new)

    # structural tf.to_float → tf.cast
    new = replace_to_float(new)

    if new != original:
        print(f"Updated: {path}")
        path.write_text(new)
