import data.BaseData as bd
import autoencoders.BaseModels as abm
import data.QEnvirontent
import reinforcements.for_stock

def experimentAE():
    window = 10

    # new
    streams1 = [("BTC", "Close", "Open", 0),
                ("BTC", "High", "Open", 0),
                ("BTC", "Low", "Open", 0),
                ("BTC", "Close", "Close", 1),
                ("BTC", "Low", "Low", 1),
                ("BTC", "High", "High", 1)
                ]
    streams2 = [("BTC", "Close", "Open", 0),
                ("BTC", "High", "Open", 0),
                ("BTC", "Low", "Open", 0),
                ("BTC", "Close", "Close", 1),
                ("BTC", "Low", "Low", 1),
                ("BTC", "High", "High", 1)
                ]
    new_data = bd.getDifData(window=window, streams=streams1)
    new_test = bd.getDifData(window=window, streams=streams2, test=True)
    abm.getAutoencoder(input_data=(new_data, new_test), encod_count=int(window * 1), window=window)

def experimentQL():

    #prepare data
    window = 10
    dif_streams1 = [
                ("GOOGL", "Close", "Open", 0),
                ("GOOGL", "High", "Open", 0),
                ("GOOGL", "Low", "Open", 0),
                ("GOOGL", "Close", "Close", 1),
                ("GOOGL", "Low", "Low", 1),
                ("GOOGL", "High", "High", 1)
                ]
    dif_data = bd.getDifData(window=window, streams=dif_streams1)
    dif_test = bd.getDifData(window=window, streams=dif_streams1, test=True)
    dif_streams1 = ("GOOGL", "Close")
    pure_data = bd.getPureData(window=window+1, stream=dif_streams1)
    pure_test = bd.getPureData(window=window+1, stream=dif_streams1, test=True)

    #get encoder
    encoder, decoder, autoencoder = abm.getAutoencoder(input_data=(dif_data, dif_test), encod_count=int(window * 2), window=window)

    #QLearning
    env = data.QEnvirontent.QEnvironment(data_y=pure_data, data_x=dif_data, test_y=pure_test, test_x=dif_test)
    qs = reinforcements.for_stock.QStock(env, dif_data, front_model=encoder)
    qs.run()
    # qs = reinforcements.for_stock.QGeneticStock(env, dif_data, front_model=encoder)
    # qs.run()

def main():
    # experimentAE()
    experimentQL()




if __name__ == "__main__":
    main()
