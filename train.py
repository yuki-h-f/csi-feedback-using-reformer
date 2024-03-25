import datetime
import os
from typing import Literal, TypeVar, Optional
from scipy.io import loadmat
import numpy as np
from typing import Tuple
import torch
from torch.nn import MSELoss
import csv
from torch.utils.data import Dataset as BaseDataset
from layers.CsiFeedbackReformerAutoEncoder import CsiFeedbackReformerAutoEncoder
from layers.CsiFeedbackTransformerAutoEncoder import CsiFeedbackTransformerAutoEncoder
from timm.scheduler import CosineLRScheduler

today = datetime.date.today()
today_directory_name = "{}{}{}".format(today.year, str(today.month).zfill(2), str(today.day).zfill(2))

MODEL_FILE_POSTFIX = "pt"
CSV_FILE_POSTFIX = "csv"

BATCH_SIZE = 200
EPOCH = 500
INDOOR_RESULT_FILENAME = "./result/{}/indoor.csv".format(today_directory_name)
OUTDOOR_RESULT_FILEnAME = "./result/{}/outdoor.csv".format(today_directory_name)

def get_transformer_model_filename(epoch: int, data_type: Literal["in", "out"], code_rate: int) -> str:
    return "{}_transformer_model_code_rate_{}_epoch_{}".format(data_type, code_rate, epoch)

def get_transformer_with_quantizer_model_filename(data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> str:
    return "{}_transformer_model_code_rate_{}_quantization_bits_{}".format(data_type, code_rate, quantization_bits)

def get_reformer_model_filename(epoch: int, data_type: Literal["in", "out"], code_rate: int) -> str:
    return "{}_reformer_model_code_rate_{}_epoch_{}".format(data_type, code_rate, epoch)

def get_reformer_with_quantizer_model_filename(data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> str:
    return "{}_reformer_model_code_rate_{}_quantization_bits_{}".format(data_type, code_rate, quantization_bits)

class Dataset(BaseDataset):
    def __init__(self, noisy, perfect) -> None:
        self.noisy = noisy
        self.perfect = perfect

    def __getitem__(self, index):
        return self.noisy[index], self.perfect[index],
    
    def __len__(self):
        return len(self.noisy)

def load_data(data_type: Literal["in", "out"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    COST_2100_DIRECTORY = "./cost2100"
    train_channel = loadmat("{}/train/DATA_Htrain{}.mat".format(COST_2100_DIRECTORY, data_type))["HT"]
    test_channel = loadmat("{}/test/DATA_Htest{}.mat".format(COST_2100_DIRECTORY, data_type))["HT"]
    validation_channel = loadmat("{}/validation/DATA_Hval{}.mat".format(COST_2100_DIRECTORY, data_type))["HT"]
    return train_channel, test_channel, validation_channel


def train_raw_transformer(epoch: int, data_type: Literal["in", "out"], code_rate: int) -> None:
    # These data have already been devided into real and imaginary part
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_transformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    train_data = np.reshape(train_data, (train_data.shape[0], 64, 32))
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    val_data = np.reshape(val_data, (val_data.shape[0], 64, 32))
    train_dataset = Dataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_data).float())
    val_dataset = Dataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_data).float())
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0")
    model = CsiFeedbackTransformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=False,
        quantization_bits=12
    )
    # without quantizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=0.00005, warmup_lr_init=0.0001, warmup_t=epoch / 10)

    # with quantizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    # scheduler = CosineLRScheduler(optimizer, t_initial=EPOCH, lr_min=0.0005, warmup_lr_init=0.001, warmup_t= EPOCH / 10)
    loss_fn = MSELoss()
    model.cuda()
    loss_fn.cuda()
    min_val_loss = float("inf")
    for _epoch in range(0, epoch):
        running_loss = 0.0
        for (input, truth) in train_data_loader:
            input = input.cuda()
            truth = truth.cuda()
            output = model(input)
            loss = loss_fn(output, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(epoch=_epoch)
        batch_loss = running_loss / BATCH_SIZE
        running_val_loss = 0.0
        with torch.no_grad():
            for (val_input, val_truth) in val_data_loader:
                val_input = val_input.cuda()
                val_truth = val_truth.cuda()
                val_output = model(val_input)
                val_loss = loss_fn(val_output, val_truth)
                running_val_loss += val_loss

        batch_val_loss = running_val_loss /BATCH_SIZE
        if min_val_loss > batch_val_loss:
            min_val_loss =  batch_val_loss
            print("updating model")
            torch.save(model.state_dict(), saved_model_path + "/" + saved_model_filename)

        print('Epoch [{0}]\t'
                    'lr: {lr:.6f}\t'
                    'Loss: {loss:.10f}\t'
                    'Validation Loss: {validation_loss:.10f}\t'
                    .format(
                    _epoch,
                    lr=optimizer.param_groups[-1]['lr'],
                    loss=batch_loss,
                    validation_loss=batch_val_loss
                    ))
    

def train_raw_reformer(epoch: int, data_type: Literal["in", "out"], code_rate: int) -> None:
    # These data have already been devided into real and imaginary part
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_reformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    train_data = np.reshape(train_data, (train_data.shape[0], 64, 32))
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    val_data = np.reshape(val_data, (val_data.shape[0], 64, 32))
    train_dataset = Dataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_data).float())
    val_dataset = Dataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_data).float())
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0")
    model = CsiFeedbackReformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=False,
        quantization_bits=12
    )
    # without quantizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=0.00005, warmup_lr_init=0.0001, warmup_t= epoch / 10)

    # with quantizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    # scheduler = CosineLRScheduler(optimizer, t_initial=EPOCH, lr_min=0.0005, warmup_lr_init=0.001, warmup_t= EPOCH / 10)
    loss_fn = MSELoss()
    model.cuda()
    loss_fn.cuda()
    min_val_loss = float("inf")

    for epoch in range(0, epoch):
        running_loss = 0.0
        for (input, truth) in train_data_loader:
            input = input.cuda()
            truth = truth.cuda()
            output = model(input)
            loss = loss_fn(output, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(epoch=epoch)
        batch_loss = running_loss / BATCH_SIZE
        running_val_loss = 0.0
        with torch.no_grad():
            for (val_input, val_truth) in val_data_loader:
                val_input = val_input.cuda()
                val_truth = val_truth.cuda()
                val_output = model(val_input)
                val_loss = loss_fn(val_output, val_truth)
                running_val_loss += val_loss

        batch_val_loss = running_val_loss /BATCH_SIZE
        if min_val_loss > batch_val_loss:
            min_val_loss =  batch_val_loss
            print("updating model")
            torch.save(model.state_dict(), saved_model_path + "/" + saved_model_filename)

        print('Epoch [{0}]\t'
                    'lr: {lr:.6f}\t'
                    'Loss: {loss:.10f}\t'
                    'Validation Loss: {validation_loss:.10f}\t'
                    .format(
                    epoch,
                    lr=optimizer.param_groups[-1]['lr'],
                    loss=batch_loss,
                    validation_loss=batch_val_loss
                    ))
    


def train_reformer_with_quantizer(epoch: int, data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> None:
    # These data have already been devided into real and imaginary part
    # previous_work_saved_model_path = get_reformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_reformer_with_quantizer_model_filename(data_type, code_rate, quantization_bits) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    train_data = np.reshape(train_data, (train_data.shape[0], 64, 32))
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    val_data = np.reshape(val_data, (val_data.shape[0], 64, 32))
    train_dataset = Dataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_data).float())
    val_dataset = Dataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_data).float())
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0")
    model = CsiFeedbackReformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=True,
        quantization_bits=quantization_bits
    )
    # model.load_state_dict(torch.load(saved_model_path + "/" + previous_work_saved_model_path))
    # without quantizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=0.00005, warmup_lr_init=0.0001, warmup_t= epoch / 10)

    # with quantizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    # scheduler = CosineLRScheduler(optimizer, t_initial=EPOCH, lr_min=0.0005, warmup_lr_init=0.001, warmup_t= EPOCH / 10)
    loss_fn = MSELoss()
    model.cuda()
    loss_fn.cuda()
    min_val_loss = float("inf")

    for epoch in range(0, epoch):
        running_loss = 0.0
        for (input, truth) in train_data_loader:
            input = input.cuda()
            truth = truth.cuda()
            output = model(input)
            loss = loss_fn(output, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(epoch=epoch)
        batch_loss = running_loss / BATCH_SIZE
        running_val_loss = 0.0
        with torch.no_grad():
            for (val_input, val_truth) in val_data_loader:
                val_input = val_input.cuda()
                val_truth = val_truth.cuda()
                val_output = model(val_input)
                val_loss = loss_fn(val_output, val_truth)
                running_val_loss += val_loss

        batch_val_loss = running_val_loss /BATCH_SIZE
        if min_val_loss > batch_val_loss:
            min_val_loss =  batch_val_loss
            print("updating model")
            torch.save(model.state_dict(), saved_model_path + "/" + saved_model_filename)

        print('Epoch [{0}]\t'
                    'lr: {lr:.6f}\t'
                    'Loss: {loss:.10f}\t'
                    'Validation Loss: {validation_loss:.10f}\t'
                    .format(
                    epoch,
                    lr=optimizer.param_groups[-1]['lr'],
                    loss=batch_loss,
                    validation_loss=batch_val_loss
                    ))

def train_with_quantizer(epoch: int, data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> None:
    # These data have already been devided into real and imaginary part
    previous_work_saved_model_path = get_transformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_transformer_with_quantizer_model_filename(data_type, code_rate, quantization_bits) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    train_data = np.reshape(train_data, (train_data.shape[0], 64, 32))
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    val_data = np.reshape(val_data, (val_data.shape[0], 64, 32))
    train_dataset = Dataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_data).float())
    val_dataset = Dataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_data).float())
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0")
    model = CsiFeedbackTransformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=True,
        quantization_bits=quantization_bits
    )
    # model.load_state_dict(torch.load(saved_model_path + "/" + previous_work_saved_model_path))
    # without quantizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch, lr_min=0.00005, warmup_lr_init=0.0001, warmup_t= epoch / 10)

    # with quantizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    # scheduler = CosineLRScheduler(optimizer, t_initial=EPOCH, lr_min=0.0005, warmup_lr_init=0.001, warmup_t= EPOCH / 10)
    loss_fn = MSELoss()
    model.cuda()
    loss_fn.cuda()
    min_val_loss = float("inf")

    for _epoch in range(0, epoch):
        running_loss = 0.0
        for (input, truth) in train_data_loader:
            input = input.cuda()
            truth = truth.cuda()
            output = model(input)
            loss = loss_fn(output, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        scheduler.step(epoch=_epoch)
        batch_loss = running_loss / BATCH_SIZE
        running_val_loss = 0.0
        with torch.no_grad():
            for (val_input, val_truth) in val_data_loader:
                val_input = val_input.cuda()
                val_truth = val_truth.cuda()
                val_output = model(val_input)
                val_loss = loss_fn(val_output, val_truth)
                running_val_loss += val_loss

        batch_val_loss = running_val_loss /BATCH_SIZE
        if min_val_loss > batch_val_loss:
            min_val_loss =  batch_val_loss
            print("updating model")
            torch.save(model.state_dict(), saved_model_path + "/" + saved_model_filename)

        print('Epoch [{0}]\t'
                    'lr: {lr:.6f}\t'
                    'Loss: {loss:.10f}\t'
                    'Validation Loss: {validation_loss:.10f}\t'
                    .format(
                    _epoch,
                    lr=optimizer.param_groups[-1]['lr'],
                    loss=batch_loss,
                    validation_loss=batch_val_loss
                    ))


        
def test_raw_transformer(epoch: int, data_type: Literal["in", "out"], code_rate: int):
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_transformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    device = torch.device("cuda:0")
    print(test_data.shape)
    model = CsiFeedbackTransformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=False,
        quantization_bits=12
    )
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(saved_model_path + "/" + saved_model_filename))
    
    test_batch_size = 2000
    output = np.zeros((test_data.shape[0], 64, 32))
    for i in range(test_data.shape[0] // test_batch_size):
        test_batch = test_data[test_batch_size * i: test_batch_size * (i + 1)]
        with torch.no_grad():
            _output = model(torch.from_numpy(test_batch).float().cuda())
        output[test_batch_size * i: test_batch_size * (i + 1)] = _output.cpu().detach().numpy().copy()
    test_data_real = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 0, :, :]
    test_data_imag = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 1, :, :]
    test_data_real = np.reshape(test_data_real, (test_data_real.shape[0], -1))
    test_data_imag = np.reshape(test_data_imag, (test_data_imag.shape[0], -1))
    test_data_C = test_data_real - 0.5 + 1j*(test_data_imag-0.5)
    output_real = np.reshape(output, (output.shape[0], 2, 32, 32))[:, 0, :, :]
    output_imag = np.reshape(output,(output.shape[0], 2, 32, 32))[:, 1, :, :]
    output_real = np.reshape(output_real, (output_real.shape[0], -1))
    output_imag = np.reshape(output_imag, (output_imag.shape[0], -1))
    output_C = output_real-0.5 + 1j*(output_imag - 0.5)
    power = np.sum(abs(test_data_C) ** 2)
    print(power.shape)
    mse = np.sum(abs(test_data_C - output_C) ** 2)
    print(mse.shape)
    nmse = np.mean(mse/power)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/" + today_directory_name, exist_ok=True)
    with open(INDOOR_RESULT_FILENAME if data_type == "in" else OUTDOOR_RESULT_FILEnAME, newline="\n", mode="a") as f:
        writer = csv.writer(f)
        writer.writerow(["Transformer", int(2048 * code_rate), "raw", 1, mse, nmse, 10 * np.log10(nmse)])


def test_raw_reformer(epoch: int, data_type: Literal["in", "out"], code_rate: int):
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    saved_model_filename = get_reformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    device = torch.device("cuda:0")
    print(test_data.shape)
    model = CsiFeedbackReformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=False,
        quantization_bits=12
    )
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(saved_model_path + "/" + saved_model_filename))
    
    test_batch_size = 2000
    output = np.zeros((test_data.shape[0], 64, 32))
    for i in range(test_data.shape[0] // test_batch_size):
        test_batch = test_data[test_batch_size * i: test_batch_size * (i + 1)]
        with torch.no_grad():
            _output = model(torch.from_numpy(test_batch).float().cuda())
        output[test_batch_size * i: test_batch_size * (i + 1)] = _output.cpu().detach().numpy().copy()
    test_data_real = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 0, :, :]
    test_data_imag = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 1, :, :]
    test_data_real = np.reshape(test_data_real, (test_data_real.shape[0], -1))
    test_data_imag = np.reshape(test_data_imag, (test_data_imag.shape[0], -1))
    test_data_C = test_data_real - 0.5 + 1j*(test_data_imag-0.5)
    output_real = np.reshape(output, (output.shape[0], 2, 32, 32))[:, 0, :, :]
    output_imag = np.reshape(output,(output.shape[0], 2, 32, 32))[:, 1, :, :]
    output_real = np.reshape(output_real, (output_real.shape[0], -1))
    output_imag = np.reshape(output_imag, (output_imag.shape[0], -1))
    output_C = output_real-0.5 + 1j*(output_imag - 0.5)
    power = np.sum(abs(test_data_C) ** 2)
    print(power.shape)
    mse = np.sum(abs(test_data_C - output_C) ** 2)
    print(mse.shape)
    nmse = np.mean(mse/power)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/" + today_directory_name, exist_ok=True)
    with open(INDOOR_RESULT_FILENAME if data_type == "in" else OUTDOOR_RESULT_FILEnAME, mode="a", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["Reformer", int(2048 * code_rate), "raw", 1, mse, nmse, 10 * np.log10(nmse)])

def test_with_quantizer(epoch: int, data_type: Literal["in", "out"], code_rate: int, quantization_bits: int, use_including_quantizer_model: bool = False):
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    if use_including_quantizer_model:
        saved_model_filename = get_transformer_with_quantizer_model_filename(data_type, code_rate, quantization_bits) + "." + MODEL_FILE_POSTFIX
    else:
        saved_model_filename = get_transformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    device = torch.device("cuda:0")
    print(test_data.shape)
    model = CsiFeedbackTransformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=True,
        quantization_bits=quantization_bits
    )
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(saved_model_path + "/" + saved_model_filename))
    
    test_batch_size = 2000
    output = np.zeros((test_data.shape[0], 64, 32))
    for i in range(test_data.shape[0] // test_batch_size):
        test_batch = test_data[test_batch_size * i: test_batch_size * (i + 1)]
        with torch.no_grad():
            _output = model(torch.from_numpy(test_batch).float().cuda())
        output[test_batch_size * i: test_batch_size * (i + 1)] = _output.cpu().detach().numpy().copy()
    test_data_real = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 0, :, :]
    test_data_imag = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 1, :, :]
    test_data_real = np.reshape(test_data_real, (test_data_real.shape[0], -1))
    test_data_imag = np.reshape(test_data_imag, (test_data_imag.shape[0], -1))
    test_data_C = test_data_real - 0.5 + 1j*(test_data_imag-0.5)
    output_real = np.reshape(output, (output.shape[0], 2, 32, 32))[:, 0, :, :]
    output_imag = np.reshape(output,(output.shape[0], 2, 32, 32))[:, 1, :, :]
    output_real = np.reshape(output_real, (output_real.shape[0], -1))
    output_imag = np.reshape(output_imag, (output_imag.shape[0], -1))
    output_C = output_real-0.5 + 1j*(output_imag - 0.5)
    power = np.sum(abs(test_data_C) ** 2)
    print(power.shape)
    mse = np.sum(abs(test_data_C - output_C) ** 2)
    print(mse.shape)
    nmse = np.mean(mse/power)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/" + today_directory_name, exist_ok=True)
    with open(INDOOR_RESULT_FILENAME if data_type == "in" else OUTDOOR_RESULT_FILEnAME, mode="a", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["Transformer" + (" train twice" if use_including_quantizer_model else ""), int(2048 * code_rate), quantization_bits, 1, mse, nmse, 10 * np.log10(nmse)])


def test_reformer_with_quantizer(epoch: int, data_type: Literal["in", "out"], code_rate: int, quantization_bits: int, use_including_quantizer_model: bool = False):
    saved_model_path = "./saved_model/" + today_directory_name
    os.makedirs(saved_model_path,  exist_ok=True)
    if use_including_quantizer_model:
        saved_model_filename = get_reformer_with_quantizer_model_filename(data_type, code_rate, quantization_bits) + "." + MODEL_FILE_POSTFIX
    else:
        saved_model_filename = get_reformer_model_filename(epoch, data_type, code_rate) + "." + MODEL_FILE_POSTFIX
    train_data, test_data, val_data = load_data(data_type)
    test_data = np.reshape(test_data, (test_data.shape[0], 64, 32))
    device = torch.device("cuda:0")
    print(test_data.shape)
    model = CsiFeedbackReformerAutoEncoder(
        num_embeddings=64,
        max_len=64,
        pad_idx=0,
        d_model=64,
        N=1,
        d_ff=128,
        heads_num=4,
        dropout_rate=0.1,
        layer_norm_eps=10e-5,
        device=device,
        code_rate=code_rate,
        use_quantizer=True,
        quantization_bits=quantization_bits
    )
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(saved_model_path + "/" + saved_model_filename))
    
    test_batch_size = 2000
    output = np.zeros((test_data.shape[0], 64, 32))
    for i in range(test_data.shape[0] // test_batch_size):
        test_batch = test_data[test_batch_size * i: test_batch_size * (i + 1)]
        with torch.no_grad():
            _output = model(torch.from_numpy(test_batch).float().cuda())
        output[test_batch_size * i: test_batch_size * (i + 1)] = _output.cpu().detach().numpy().copy()
    test_data_real = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 0, :, :]
    test_data_imag = np.reshape(test_data, (test_data.shape[0], 2, 32, 32))[:, 1, :, :]
    test_data_real = np.reshape(test_data_real, (test_data_real.shape[0], -1))
    test_data_imag = np.reshape(test_data_imag, (test_data_imag.shape[0], -1))
    test_data_C = test_data_real - 0.5 + 1j*(test_data_imag-0.5)
    output_real = np.reshape(output, (output.shape[0], 2, 32, 32))[:, 0, :, :]
    output_imag = np.reshape(output,(output.shape[0], 2, 32, 32))[:, 1, :, :]
    output_real = np.reshape(output_real, (output_real.shape[0], -1))
    output_imag = np.reshape(output_imag, (output_imag.shape[0], -1))
    output_C = output_real-0.5 + 1j*(output_imag - 0.5)
    power = np.sum(abs(test_data_C) ** 2)
    print(power.shape)
    mse = np.sum(abs(test_data_C - output_C) ** 2)
    print(mse.shape)
    nmse = np.mean(mse/power)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/" + today_directory_name, exist_ok=True)
    with open(INDOOR_RESULT_FILENAME if data_type == "in" else OUTDOOR_RESULT_FILEnAME, mode="a", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["Reformer" + (" train twice" if use_including_quantizer_model else ""), int(2048 * code_rate), quantization_bits, 1, mse, nmse, 10 * np.log10(nmse)])

if __name__ == "__main__":
    code_rate_list = [1, 1 / 2 , 1 / 4, 1 / 8, 1 / 16, 1/ 32]
    quantization_bits_list =[4, 6, 8]
    data_types = ["out", "in"]
    
    for data_type in data_types:
        for code_rate in code_rate_list:
            # train_raw_reformer(EPOCH, data_type, code_rate)
            # test_raw_reformer(EPOCH, data_type, code_rate)
            # train_raw_transformer(EPOCH, data_type, code_rate)
            # test_raw_transformer(EPOCH, data_type, code_rate)
            for quantization_bits in quantization_bits_list:
                train_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits)
                test_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits, True)
                # train_reformer_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits)
                # test_reformer_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits, True)
     
    # for data_type in data_types:
    #     for code_rate in code_rate_list:
    #         train_raw_transformer(EPOCH * 2, data_type, code_rate)
    #         train_raw_reformer(EPOCH * 2, data_type, code_rate)
    #         test_raw_transformer(EPOCH * 2, data_type, code_rate)
    #         test_raw_reformer(EPOCH * 2, data_type, code_rate)
    #         for quantization_bits in quantization_bits_list:
    #             test_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits)
    #             test_reformer_with_quantizer(EPOCH * 2, data_type, code_rate, quantization_bits)   