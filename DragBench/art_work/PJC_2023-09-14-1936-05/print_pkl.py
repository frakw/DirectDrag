import pickle

def load_and_print_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("內容:")
            print(data)
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")

if __name__ == "__main__":
    file_path = input("請輸入 .pkl 檔案路徑: ")
    load_and_print_pkl(file_path)

