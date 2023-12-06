import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):

    def __init__(self, batch_size, input_encoding_size, hidden_size, output_size, device):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.input_embedding = nn.Linear(input_encoding_size, hidden_size)
        self.lstm_bottom = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first = True,
                                 bidirectional = True)
        self.skip_connection_one = nn.Linear(hidden_size * 2, hidden_size)  
        self.lstm_middle = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 num_layers=2,
                                 batch_first = True,
                                 bidirectional = True)
        self.skip_connection_two = nn.Linear(hidden_size * 2, hidden_size)  
        self.lstm_top = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 num_layers=1,
                                 batch_first = True,
                                 bidirectional = True)
        self.final_linear = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, inputs, hidden_and_cell_states, temperature):
        
        inputs = self.input_embedding(inputs)
        output_bottom, hidden_and_cell_states_bottom = self.lstm_bottom(inputs, hidden_and_cell_states[0])
        skip_out_one = self.skip_connection_one(output_bottom) + inputs
        output_middle, hidden_and_cell_states_middle = self.lstm_middle(skip_out_one, hidden_and_cell_states[1])
        skip_out_two = self.skip_connection_two(output_middle) + inputs
        output_top, hidden_and_cell_states_top = self.lstm_top(skip_out_two, hidden_and_cell_states[2])
        output = self.final_linear(output_top)
        
        gumbel_noise = -torch.empty_like(output).exponential_().log() * 0.5 
        one_hot_output = torch.nn.functional.gumbel_softmax(output + gumbel_noise, tau=temperature, hard=True)
        return one_hot_output, (hidden_and_cell_states_bottom, hidden_and_cell_states_middle, hidden_and_cell_states_top)
      
    def init_zero_state(self):
        init_hidden_bottom = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_cell_bottom = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_hidden_middle = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_cell_middle = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_hidden_top = torch.zeros(2, self.batch_size, self.hidden_size).to(self.device)
        init_cell_top = torch.zeros(2, self.batch_size, self.hidden_size).to(self.device)
        return (init_hidden_bottom, init_cell_bottom), (init_hidden_middle, init_cell_middle), (init_hidden_top, init_cell_top)


class LSTMDiscriminator(nn.Module):

    def __init__(self, batch_size, input_encoding_size, hidden_size, device):
        super().__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.input_embedding = nn.Linear(input_encoding_size, hidden_size)
        self.lstm_bottom = nn.LSTM(input_size=hidden_size, 
                           hidden_size=hidden_size, 
                           num_layers=2,
                           batch_first=True, 
                           bidirectional=True)
        self.skip_connection_one = nn.Linear(hidden_size * 2, hidden_size)  
        self.lstm_middle = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size, 
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        self.skip_connection_two = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm_top = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size, 
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.char_frequency_tracker = nn.Linear(input_encoding_size, 50)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.final_linear = nn.Linear(hidden_size * 10 + 50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        
        hidden_and_cell_states_bottom, hidden_and_cell_states_middle, hidden_and_cell_states_top = self._init_zero_state()
        
        input_char_frequencies = torch.sum(inputs, dim=1)
        frequencies_transformed = self.leaky_relu(self.char_frequency_tracker(input_char_frequencies))
        
        inputs = self.input_embedding(inputs)
        output_bottom, (hidden_states_bottom, _) = self.lstm_bottom(inputs, hidden_and_cell_states_bottom) 
        skip_out_one = self.skip_connection_one(output_bottom) + inputs
        output_middle, (hidden_states_middle, _) = self.lstm_middle(skip_out_one, hidden_and_cell_states_middle) 
        skip_out_two = self.skip_connection_two(output_middle) + inputs
        _, (hidden_states_top, _) = self.lstm_top(skip_out_two, hidden_and_cell_states_top)
        
        hidden_states_bottom = hidden_states_bottom.permute(1, 0, 2)
        hidden_states_middle = hidden_states_middle.permute(1, 0, 2)
        hidden_states_top = hidden_states_top.permute(1, 0, 2)
        hidden_states_concatenated = torch.cat((hidden_states_bottom, hidden_states_middle, hidden_states_top), dim=1)
        hidden_states_concatenated = hidden_states_concatenated.reshape(hidden_states_concatenated.size(0), -1)
        
        hidden_states_and_frequencies = torch.cat((hidden_states_concatenated, frequencies_transformed), dim=1)
        
        logit = self.final_linear(hidden_states_and_frequencies)
        output = self.sigmoid(logit)
        return output
    
    def _init_zero_state(self):
        init_hidden_bottom = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_cell_bottom = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_hidden_middle = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_cell_middle = torch.zeros(4, self.batch_size, self.hidden_size).to(self.device)
        init_hidden_top = torch.zeros(2, self.batch_size, self.hidden_size).to(self.device)
        init_cell_top = torch.zeros(2, self.batch_size, self.hidden_size).to(self.device)
        return (init_hidden_bottom, init_cell_bottom), (init_hidden_middle, init_cell_middle), (init_hidden_top, init_cell_top)
