import json
import logging
import matplotlib.pyplot as plt
import string
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from model import LSTMDiscriminator, LSTMGenerator


def initialize_standard_settings():
    global RANDOM_SEED
    global DEVICE
    global BATCH_SIZE
    global HIDDEN_DIM
    global GEN_LEARNING_RATE
    global DISC_LEARNING_RATE
    global TEMPERATURE
    global CHARACTER_SET
    
    RANDOM_SEED = 789
    DEVICE = torch.device('cpu')
    
    BATCH_SIZE = 32
    HIDDEN_DIM = 512

    GEN_LEARNING_RATE = 5e-5
    DISC_LEARNING_RATE = 5e-5
    TEMPERATURE = 2
    
    CHARACTER_SET = string.printable[:-3]  # specifically excluding \r, \x0b, \x0c
    
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO)


def create_dataloaders():
    with open('/data/castelobranco_truncated.txt', 'r') as file:
        cb = [line for line in file.readlines() if line.strip()]
    with open('/data/pessoa_truncated.txt', 'r') as file:
        pe = [line for line in file.readlines() if line.strip()]
    with open('/data/shakespeare_truncated.txt', 'r') as file:
        sh = [line for line in file.readlines() if line.strip()]

    cb_tensor = torch.zeros((len(cb), 28, len(CHARACTER_SET)), dtype=torch.int)
    pe_tensor = torch.zeros((len(cb), 28, len(CHARACTER_SET)), dtype=torch.int)
    sh_tensor = torch.zeros((len(cb), 28, len(CHARACTER_SET)), dtype=torch.int)

    for ind, line in enumerate(cb):
        for pos, char in enumerate(line):
            cb_tensor[ind][pos][CHARACTER_SET.index(char)] = 1
    for ind, line in enumerate(pe):
        for pos, char in enumerate(line):
            pe_tensor[ind][pos][CHARACTER_SET.index(char)] = 1
    for ind, line in enumerate(sh):
        for pos, char in enumerate(line):
            sh_tensor[ind][pos][CHARACTER_SET.index(char)] = 1
    
    cb_dataloader = DataLoader(TensorDataset(cb_tensor), batch_size=BATCH_SIZE, shuffle=True)
    pe_dataloader = DataLoader(TensorDataset(pe_tensor), batch_size=BATCH_SIZE, shuffle=True)
    sh_dataloader = DataLoader(TensorDataset(sh_tensor), batch_size=BATCH_SIZE, shuffle=True)

    return {'castelobranco': cb_dataloader, 'pessoa': pe_dataloader, 'shakespeare': sh_dataloader}


def save_models():
    pass


def evaluate_generators():
    pass


def main():

    initialize_standard_settings()
    dataloaders = create_dataloaders()

    with open(sys.argv[1], 'r') as file:
        training_scheme = json.load(file)

    generators = {}
    discriminators = {}
    generator_optimizers = {}
    discriminator_optimizers = {}
    generator_losses = {}
    discriminator_losses = {}

    for generator in training_scheme['generators']:
        generators[generator] = LSTMGenerator(BATCH_SIZE, len(CHARACTER_SET), HIDDEN_DIM, len(CHARACTER_SET), DEVICE).to(DEVICE)
        generator_optimizers[generator] = torch.optim.Adam(generators[generator].parameters(), lr=GEN_LEARNING_RATE)
        generator_losses[generator] = []
        logging.info(f"Initialized generator '{generator}' with optimizer and learning rate {GEN_LEARNING_RATE}")
    for discriminator in training_scheme['discriminators']:
        discriminators[discriminator] = LSTMDiscriminator(BATCH_SIZE, len(CHARACTER_SET), HIDDEN_DIM, DEVICE).to(DEVICE)
        discriminator_optimizers[discriminator] = torch.optim.Adam(discriminators[discriminator].parameters(), lr=DISC_LEARNING_RATE)
        discriminator_losses[discriminator] = []
        logging.info(f"Initialized discriminator '{discriminator}' with optimizer and learning rate {DISC_LEARNING_RATE}")
    criterion = torch.nn.BCELoss()

    for phase in training_scheme['training_phases']:
        phase_generator_losses = []
        phase_discriminator_losses = []

        generator_id = phase['generator']
        discriminator_id = phase['discriminator']
        logging.info(f"Starting phase {phase['order']}: {phase['epochs']} epochs on {phase['data']}")
        logging.info(f"Involved models: {generator_id}, {discriminator_id}")
        
        generators[generator_id].train()
        discriminators[discriminator_id].train()

        for epoch in trange(phase['epochs']):
            
            for batch in dataloaders[phase['data']]:
            
                # GENERATOR TRAINING
                generator_optimizers[generator_id].zero_grad()
                generator_inputs = torch.randn(BATCH_SIZE, 1, len(CHARACTER_SET)).to(DEVICE)
                generator_hidden_and_cell = generators[generator_id].init_zero_state()
                generated_sequences = None
            
                for _ in range(28):
                    generator_outputs, generator_hidden_and_cell = generators[generator_id](generator_inputs, generator_hidden_and_cell, TEMPERATURE)
                    if not generated_sequences:
                        generated_sequences = generator_outputs
                    else:
                        generated_sequences = torch.cat((generated_sequences, generator_outputs), dim=1)
                    generator_inputs = generator_outputs
                
                judgments = discriminators[discriminator_id](generated_sequences)
                labels = torch.ones_like(judgments)
                generator_loss = criterion(judgments, labels)
                generator_loss.backward()
                generator_optimizers[generator_id].step()

                phase_generator_losses.append(generator_loss.item())
                
                # DISCRIMINATOR TRAINING
                discriminator_optimizers[discriminator_id].zero_grad()
                batch = batch.to(DEVICE)
                
                judgments_on_real = discriminators[discriminator_id](batch)
                labels_for_real = torch.ones_like(judgments_on_real)
                discriminator_loss_real = criterion(judgments_on_real, labels_for_real)

                judgments_on_generated = discriminators[discriminator_id](generated_sequences.detach())
                labels_for_generated = torch.zeros_like(judgments_on_generated)
                discriminator_loss_generated = criterion(judgments_on_generated, labels_for_generated)
                
                discriminator_loss = (discriminator_loss_real + discriminator_loss_generated) / 2
                discriminator_loss.backward()
                discriminator_optimizers[discriminator_id].step()

                phase_discriminator_losses.append(discriminator_loss.item())
                
            if epoch % 5 == 0:
                logging.info(f'End of Epoch {epoch}: Generator Loss {generator_loss.item():.3f}, Discriminator Loss {discriminator_loss.item():.3f}')
                # print out 8 examples of generator output for reference
                for ind in range(8):
                    predicted = ''
                    for one_hot in generated_sequences[ind]:
                        predicted += CHARACTER_SET[torch.nonzero(one_hot).item()]
                    logging.info(f'Example output {ind}: {predicted}')

        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.plot(range(len(phase_generator_losses)), phase_generator_losses)
        plt.plot(range(len(phase_discriminator_losses)), phase_discriminator_losses)
        plt.legend((generator_id, discriminator_id))
        plt.savefig(f"/plots/scheme_{training_scheme['id']}_phase_{phase['order']}.png")

        generator_losses[generator_id].extend(phase_generator_losses)
        discriminator_losses[discriminator_id].extend(phase_discriminator_losses)
        logging.info(f"Phase {phase['order']} complete")
    
    save_models()
    evaluate_generators()
    

if __name__=="__main__":
    main()
