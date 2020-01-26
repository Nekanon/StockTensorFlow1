import data.BaseData as bd
import autoencoders.BaseModels as abm

def main():
    pure_data = bd.getData()
    abm.experimental1(input_data = pure_data)

if __name__=="__main__":
    main()