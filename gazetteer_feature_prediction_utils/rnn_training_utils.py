import torch
import os
import datetime


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss,
                  'char_dictionary': model.char_dictionary,
                  'output_size': model.output_size,
                  'device': model.device,
                  'char_embedding_dim': model.char_embedding_dim,
                  'hidden_size_char': model.hidden_size_char
                  }

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    # TODO: hier müssten irgendwie noch die **kwargs aus dem state_dict für die model init geladen werden

    model.load_state_dict(state_dict['model_state_dict'])
    #optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def train(model,
          optimizer,
          criterion,
          train_loader,
          dev_loader,
          test_loader,
          file_path,
          eval_every = 100,
          num_epochs=5,
          best_valid_loss=float("Inf"),
          ):

    # create path if not there
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            model.train()

            output, target_vectors = model(batch)
            #output, target_vectors, weight_info = model(batch)

            #print(output, target_vectors)

            loss = criterion(output, target_vectors)

            #### hier eine Gewichtung einbauen
            # also je nach coonfidence / tagged-untagged Loss multiplizieren, damit manche mehr/weniger ins Gewicht fallsen
            # siehe train_model_deprecated.py
            # TODO: Achtung: so würde das aktuell dazu führen, dass die random spans NICHT gewertet werden, weil die [0,0] als conf und ratio haben
            # das müsste man also ändern. Wie? Wenn ich die einfach [1.0, 1.0] setze, werden sie vergleichsweise stark gewichtet, weil die "echteh"
            # ja < 1.0 haben. Wäre das schlimm?

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            print(f"At global_step (batch) {global_step}, loss in batch is {round(loss.item(), 2)}", end="\r")
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for batch in dev_loader:
                        #output, target_vectors, weight_info = model(batch)
                        output, target_vectors = model(batch)

                        loss = criterion(output, target_vectors)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(dev_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('{}:\tEpoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}\n'
                                   .format(datetime.datetime.now(), epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                           average_train_loss, average_valid_loss))

                # write progress to log:

                with open(file_path + "/log.txt", "a") as log_file:
                    log_file.write('{}:\tEpoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}\n'
                                   .format(datetime.datetime.now(), epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                           average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
