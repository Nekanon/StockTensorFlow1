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
    abm.experimental3(input_data=(new_data, new_test), encod_count=int(window * 1), window=window)

def experimentQL():
    window = 10
    dif_streams1 = [
                ("C", "Close", "Open", 0),
                ("C", "High", "Open", 0),
                ("C", "Low", "Open", 0),
                ("C", "Close", "Close", 1),
                ("C", "Low", "Low", 1),
                ("C", "High", "High", 1)
                ]
    dif_data = bd.getDifData(window=window, streams=dif_streams1)
    dif_streams1 = ("C", "Close")
    pure_data = bd.getPureData(window=window+2, stream=dif_streams1)
    env = data.QEnvirontent.QEnvironment(pure_data, dif_data)
    qs = reinforcements.for_stock.QStock(env, dif_data)
    qs.run()


def main():
    # experimentAE()
    experimentQL()




if __name__ == "__main__":
    main()
