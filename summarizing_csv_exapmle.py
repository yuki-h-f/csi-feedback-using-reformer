from typing import Literal
EPOCH = 1000


def get_transformer_model_filename(data_type: Literal["in", "out"], code_rate: int) -> str:
    return "{}_transformer_model_epochs_{}_code_rate_{}".format(data_type, EPOCH, code_rate)

def get_transformer_with_quantizer_model_filename(data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> str:
    return "{}_transformer_model_epochs_{}_code_rate_{}_quantization_bits_{}".format(data_type, EPOCH, code_rate, quantization_bits)

def get_reformer_model_filename(data_type: Literal["in", "out"], code_rate: int) -> str:
    return "{}_reformer_model_epochs_{}_code_rate_{}".format(data_type, EPOCH, code_rate)

def get_reformer_with_quantizer_model_filename(data_type: Literal["in", "out"], code_rate: int, quantization_bits: int) -> str:
    return "{}_reformer_model_epochs_{}_code_rate_{}_quantization_bits_{}".format(data_type, EPOCH, code_rate, quantization_bits)

if __name__ == "__main__":
    code_rate_list = [1, 1 / 2 , 1 / 4, 1 / 8, 1 / 16, 1/ 32]
    quantization_bits_list =[4, 6, 8]
    data_types = ["out", "in"]
    for data_type in data_types:
        for quantization_bit in quantization_bits_list:

            with open("./summary/" + data_type + "_" + str(EPOCH) + "quantizatin_bits_" + str(quantization_bit) + ".csv", "a") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["overhead", "transformer", "transformer from scratch", "reformer", "reformer from scratch"])
                for code_rate in code_rate_list:
                    csv_row = [str(2048 * code_rate)]
                    with open(get_transformer_with_quantizer_model_filename(data_type, code_rate, quantization_bit) + ".csv", "r") as f2:
                        reader = csv.reader(f2)
                        rows = [row for row in reader]
                        nmsedb = rows[0][3]
                        csv_row.append(nmsedb)
                    with open(get_transformer_with_quantizer_model_filename(data_type, code_rate, quantization_bit) + "_from_non_quantized.csv", "r") as f2:
                        reader = csv.reader(f2)
                        rows = [row for row in reader]
                        nmsedb = rows[0][3]
                        csv_row.append(nmsedb)
                    with open(get_reformer_with_quantizer_model_filename(data_type, code_rate, quantization_bit) + ".csv", "r") as f2:
                        reader = csv.reader(f2)
                        rows = [row for row in reader]
                        nmsedb = rows[0][3]
                        csv_row.append(nmsedb)
                    with open(get_reformer_with_quantizer_model_filename(data_type, code_rate, quantization_bit) + "_from_non_quantized.csv", "r") as f2:
                        reader = csv.reader(f2)
                        rows = [row for row in reader]
                        nmsedb = rows[0][3]
                        csv_row.append(nmsedb)
                    writer.writerow(csv_row)



