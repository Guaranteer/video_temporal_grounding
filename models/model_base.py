import torch
import torch.nn as nn





class Model(nn.Module):
    def __init__(self, params, device):
        super(Model, self).__init__()

        self.params = params
        self.device = device
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']

        self.conv_bottom = nn.Sequential(
            nn.Conv1d(self.input_video_dim, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_64 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=64, stride=16),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_128 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=128, stride=32),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_256 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=256, stride=64),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_encoder_512 = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=512, stride=128),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.conv_top = nn.Sequential(
            nn.Conv1d(self.hidden_size,self.hidden_size,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU())

        self.ques_rnn = nn.GRU(self.input_ques_dim, self.hidden_size, batch_first=True)


        self.fc_base = nn.Linear(self.hidden_size * 4 , self.hidden_size)
        self.fc_score = nn.Linear(self.hidden_size, 1)
        self.fc_reg = nn.Linear(self.hidden_size, 2)

    def forward(self, frame_vecs, frame_n, ques_vecs, ques_n, labels, regs, idxs):

        frame_vecs = torch.transpose(frame_vecs, 1, 2)
        conv_bottom_out = self.conv_bottom(frame_vecs)
        conv_encoder_64_out = self.conv_encoder_64(conv_bottom_out)
        conv_encoder_128_out = self.conv_encoder_128(conv_bottom_out)
        conv_encoder_256_out = self.conv_encoder_256(conv_bottom_out)
        conv_encoder_512_out = self.conv_encoder_512(conv_bottom_out)
        conv_top_64_out = self.conv_top(conv_encoder_64_out)
        conv_top_128_out = self.conv_top(conv_encoder_128_out)
        conv_top_256_out = self.conv_top(conv_encoder_256_out)
        conv_top_512_out = self.conv_top(conv_encoder_512_out)
        conv_encoder_out = torch.cat((conv_top_64_out, conv_top_128_out, conv_top_256_out, conv_top_512_out),dim=2)
        conv_encoder_out = torch.transpose(conv_encoder_out, 1, 2)

        # Set initial hidden and cell states

        _, idx_sort = torch.sort(ques_n, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        ques = ques_vecs.index_select(0, idx_sort)
        lengths = list(ques_n[idx_sort])
        ques_packed = nn.utils.rnn.pack_padded_sequence(input=ques, lengths=lengths, batch_first=True)

        h0 = torch.zeros(1, ques_vecs.size(0), self.hidden_size).to(self.device)
        ques_padded, ques_hidden = self.ques_rnn(ques_packed, h0)

        ques_padded = nn.utils.rnn.pad_packed_sequence(ques_padded, batch_first=True)
        ques_out = ques_padded[0].index_select(0, idx_unsort)
        ques_hidden = ques_hidden.squeeze(0)
        ques_hidden = ques_hidden.unsqueeze(1).expand(-1, conv_encoder_out.size(1), self.hidden_size)

        fused_add = ques_hidden + conv_encoder_out
        fused_mul = ques_hidden * conv_encoder_out
        fused_cat = torch.cat((ques_hidden, conv_encoder_out), dim = 2)
        fused_all = torch.cat((fused_add, fused_mul, fused_cat), dim = 2)

        fused_all = self.fc_base(fused_all)
        score = self.fc_score(fused_all)
        score = score.squeeze(2)
        predict_reg = self.fc_reg(fused_all)

        flag = (labels - 0.5) * (-2)
        score_loss = torch.log(1 + torch.exp(flag * score)) # [45, 21, 9, 3]
        raw_index = torch.range(0, ques_vecs.size(0), dtype=torch.int32)
        pos_loss = score_loss[raw_index, idxs]
        all_score_loss = torch.sum(score_loss) / 78 + torch.sum(pos_loss)



        pass



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)