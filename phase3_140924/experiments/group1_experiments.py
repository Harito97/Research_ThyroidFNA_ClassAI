

def get_data(data_dir:str):
    ...

def get_model(name:str):
    ...

def train_model_with_custom_loss_func(model, loss_func):
    ...

def test_model(model, data):
    ...

def run_experiment(data_dir:str, model_name:str):
    data = get_data(data_dir)
    model = get_model(model_name)
    train_model_with_custom_loss_func(model, loss_func)
    test_model(model, data)

