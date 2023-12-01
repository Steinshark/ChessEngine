import torch 
import chess 
import numpy
import time
from model import ChessModel
from torch.utils.data import Dataset,DataLoader
import random 
import math 

class chessDataSet(Dataset):

    def __init__(self,boards,results):

        self.data   = [] 

        for b,r in zip(boards,results):

            self.data.append([b,r])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def fen_to_tensor(fen_list):
    #fen_list        = [fen.split(' ')[0] for fen in fen_list]
    orientation     = 1 if chess.Board().turn else -1
    batch_size 		= len(fen_list)
    board_tensors 	= numpy.zeros(shape=(batch_size,7,8,8),dtype=numpy.float32)

    piece_indx 		= {"R":4,"N":8,"B":6,"Q":2,"K":0,"P":10,"r":5,"n":9,"b":7,"q":3,"k":1,"p":11}

    pieces          = {"R":1*orientation,"N":1*orientation,"B":1*orientation,"Q":1*orientation,"K":1*orientation,"P":1*orientation,"r":-1*orientation,"n":-1*orientation,"b":-1*orientation,"q":-1*orientation,"k":-1*orientation,"p":-1*orientation}

    #Go through FEN and fill pieces

    for i in range(len(fen_list)):
        for j in range(1,9):
            fen_list[i] 	= fen_list[i].replace(str(j),"e"*j)

    for i,fen in enumerate(fen_list):
        try:
            position	= fen.split(" ")[0].split("/")
            turn 		= fen.split(" ")[1]
            castling 	= fen.split(" ")[2]
        except IndexError:
            print(f"weird fen: {fen}")
        
        #Place pieces
        for rank_i,rank in enumerate(reversed(position)):
            for file_i,piece in enumerate(rank): 
                if not piece == "e":
                    channel                             = int(piece_indx[piece] / 2)
                    board_tensors[i,channel,rank_i,file_i]	= pieces[piece]
        
    return torch.from_numpy(board_tensors)

def play_game(model:ChessModel,max_ply=256):

    game_boards         = [] 
    game_outputs        = [] 


    board               = chess.Board() 

    while (not board.is_game_over()) and (not board.ply() > max_ply):

        #Get all moves 
        all_moves       = list(board.generate_legal_moves())
        new_boards      = [board.copy(stack=False) for move in all_moves]
        [board.push(move) for board,move in zip(new_boards,all_moves)]

        with torch.no_grad():
            all_boards              = fen_to_tensor([b.fen() for b in new_boards]).to(torch.device('cuda'))
            move_outs               = model.forward(all_boards).cpu().numpy() * -1                                      #Scores were calculated from Other perspective. AkA we need to pick argmin
            corrected_move_outs     = [math.e**weight for weight in move_outs] 
            corrected_move_outs     = corrected_move_outs / max(corrected_move_outs)                                    #Scale back to 1
            chosen_move             = random.choices(all_moves,corrected_move_outs,k=1)[0]

        #Save states 
        game_boards.append(fen_to_tensor([board.fen()])[0])
        game_outputs.append(1 if board.turn else -1)
        board.push(chosen_move)



    result          = board.result()


    if result == "1-0":
        pass
    elif result == "0-1":
        game_outputs = [i*-1 for i in game_outputs]
    else:
        game_outputs = [0 for _ in game_outputs]

    return game_boards,game_outputs


def play_graded_game(white_model:ChessModel,black_model:ChessModel,max_ply=512):

    board               = chess.Board() 
    r_n                 = 14

    while board.ply() < max_ply:

        randomizing     = int(board.ply() < r_n) * (.05*(r_n-board.ply())/r_n)
        #Make white's move 
        
        #Get all moves 
        all_moves       = list(board.generate_legal_moves())
        new_boards      = [board.copy(stack=False) for move in all_moves]
        [board.push(move) for board,move in zip(new_boards,all_moves)]

        #Calculate move
        with torch.no_grad():
            all_boards              = fen_to_tensor([b.fen() for b in new_boards]).to(torch.device('cuda'))
            move_outs               = white_model.forward(all_boards).cpu().numpy() * -1                        #Scores were calculated from Other perspective. AkA we need to pick argmin
            corrected_move_outs     = [math.e**weight for weight in move_outs]
            corrected_move_outs     = numpy.array(corrected_move_outs) / max(1,max(corrected_move_outs))        #Clip to 1 
            #If below move 10, add noise 
            if randomizing:
                #print(f"move {board.ply()} rand={randomizing}")
                corrected_move_outs = [weight+(randomizing*random.random()*max(corrected_move_outs)) for weight in move_outs]
            max_move_i              = numpy.argmax(corrected_move_outs)
            chosen_move             = all_moves[max_move_i]
        
        #Play move
        board.push(chosen_move)

        #check end of game
        if board.is_game_over():
            result          = board.result()
            if result == "1-0":
                return 1 
            elif result == "0-1":
                return -1
            else:
                return 0



        #Make black's move 
        
        #Get all moves 
        all_moves       = list(board.generate_legal_moves())
        new_boards      = [board.copy(stack=False) for move in all_moves]
        [board.push(move) for board,move in zip(new_boards,all_moves)]

        #Calculate move
        with torch.no_grad():
            all_boards              = fen_to_tensor([b.fen() for b in new_boards]).to(torch.device('cuda'))
            move_outs               = black_model.forward(all_boards).cpu().numpy() * -1                        #Scores were calculated from Other perspective. AkA we need to pick argmin
            corrected_move_outs     = [math.e**weight for weight in move_outs] 
            corrected_move_outs     = numpy.array(corrected_move_outs) / max(1,max(corrected_move_outs))        #Clip to 1 
            #If below move 10, add noise 
            if randomizing:
                #print(f"move {board.ply()} rand={randomizing}")
                corrected_move_outs = [weight+(randomizing*random.random()) for weight in move_outs]
            max_move_i              = numpy.argmax(corrected_move_outs)
            chosen_move             = all_moves[max_move_i]
        
        #Play move
        board.push(chosen_move)

        #check end of game
        if board.is_game_over():
            result          = board.result()
            if result == "1-0":
                return 1 
            elif result == "0-1":
                return -1
            else:
                return 0


def play_championship(model1:ChessModel,model2:ChessModel,n_games=100,max_ply=512):
    

    game_outcomes   = {"model1":0,"model2":0,"draw":0}

    #Play half of games as w/b
    for game_num in range(int(n_games/2)):
        outcome     = play_graded_game(model1,model2,max_ply)

        if outcome == 1:
            game_outcomes["model1"]     += 1
        elif outcome == -1:
            game_outcomes["model2"]     += 1
        else:
            game_outcomes["draw"]       += 1
    
    #Play half of games as b/w
    for game_num in range(int(n_games/2)):
        outcome     = play_graded_game(model2,model1,max_ply)

        if outcome == 1:
            game_outcomes["model2"]     += 1
        elif outcome == -1:
            game_outcomes["model1"]     += 1
        else:
            game_outcomes["draw"]       += 1

    
    return game_outcomes
    



if __name__ == "__main__":


    models          = {i:ChessModel(7).to(torch.device('cuda')) for i in range(4)}
    training_i      = 0
    training_model  = models[training_i] 
     
    t0 = time.time()

    iters       = 1024 
    n_games     = 32
    bs          = 32

    
    for iter in range(iters):
        print(f"\n\nTRAINING EPOCH {iter}\n")
        boards      = [] 
        results     = []
        losses      = [] 

        #Replace current best model and prep for eval 
        training_model  = models[training_i] 
        training_model   = training_model.eval()
        print(f"\ttraining with model {training_i}")

        #Play training games
        for _ in range(n_games):
            b,r     = play_game(training_model,400)
            for b_,r_ in zip(b,r):
                boards.append(b_)
                results.append(r_)
        print(f"\tTraining\tn_exps: {len(boards)}\t\n\t\t\t    qf: {100*(1-(results.count(0)/len(results))):.2f}%")

        #Train 
        cd  = chessDataSet(boards,results)

        training_model   = training_model.train()
        optim       = torch.optim.Adam(training_model.parameters(),lr=.0002)
        loss_fn     = torch.nn.MSELoss()
        for i,batch in enumerate(DataLoader(cd,batch_size=bs,shuffle=True)):
            
            for p in training_model.parameters():
                p.grad  = None 

            gamestate       = batch[0].to(torch.device('cuda')).type(torch.float32)
            score           = batch[1].to(torch.device('cuda')).type(torch.float32).unsqueeze(dim=-1)

            predicted   = training_model.forward(gamestate)
            #print(f"scores={score.shape}: {score[:10]}\npredicted={predicted.shape}: {predicted[:10]}")

            loss        = loss_fn(predicted,score)
            loss.backward()
            losses.append(loss.mean().item())

            optim.step()
        print(f"\t\t\t  loss: {sum(losses)/len(losses):.5f}\n")

        #Check next best
        print(f"\tRunning pairings")
        pairings    = [[training_i,x] for x in models if not x == training_i]
        top_wr      = 0 
        wr_thresh   = .05
        for pair in pairings:
            n_games             = 100

            model1              = models[pair[0]].eval()
            model2              = models[pair[1]].eval()

            pair_outcome        = play_championship(model1,model2,n_games=n_games)
            model1_wins         = pair_outcome['model1']
            model2_wins         = pair_outcome['model2']
            draws               = pair_outcome['draw']
            wr1                 = model1_wins/(model1_wins+model2_wins+.001)
            wr2                 = model2_wins/(model1_wins+model2_wins+.001)

            #If model1 wins > 10% of time and has > 60% winrate over other, its better
            if model1_wins > wr_thresh*n_games and wr1 > .6:

                #Replace best
                if wr1 > top_wr:
                    top_wr      = wr1
                    training_i  = pair[0]
            
            #If model1 wins > 10% of time and has > 60% winrate over other, its better
            if model2_wins > .1*n_games and wr2 > .6:

                #Replace best
                if wr2 > top_wr:
                    top_wr      = wr2 
                    training_i  = pair[1]


            print(f"\t\tPlayed model{pair[0]} vs. model{pair[1]}: {pair_outcome}\n\t\twr1 was {wr1}\ttop wr={top_wr}\ttop model={training_i}\n")



            

            

