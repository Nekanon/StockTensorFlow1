import data.BaseData as bd
import autoencoders.BaseModels as abm


def main():
    window = 10

    #new
    streams1 = [("NEE", "Close", "Close", 1),
               ("NEE", "High", "Open", 0),
               ("NEE", "Low", "Open", 0),
               # ("GOOGL", "Low", "Low", 1),
               # ("GOOGL", "High", "High", 1)
               ]
    streams2 = [("NEE", "Close", "Close", 1),
               ("NEE", "High", "Open", 0),
               ("NEE", "Low", "Open", 0),
               # ("GOOGL", "Low", "Low", 1),
               # ("GOOGL", "High", "High", 1)
               ]
    new_data = bd.getDataN1(window=window, streams=streams1)
    new_test = bd.getDataN1(window=window, streams=streams2, test=True)
    abm.experimental3(input_data=(new_data, new_test), encod_count=10, window=window)


    #old
    # pure_data = bd.getData("GOOGL", tiker_test="GOOGL", window=window)
    # abm.experimental1(input_data=pure_data, encod_count=6, window=window)


if __name__ == "__main__":
    main()
