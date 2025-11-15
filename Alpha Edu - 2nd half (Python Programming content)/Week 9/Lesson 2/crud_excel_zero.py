import pandas as pd

class ExcelCRUD:

    def __init__(self, file_name):
        self.file_name = file_name

    def create_excel(self):
        initial_data = {
            "Name": ["Botakoz", "Forest", "Dante", "Jerry"],
            "Age":[18, 24, 22, 26],
            "Grade":['A', "C", "A+", "D"],
            "Salary":[750, 900, 300, 100],
        }
        original_dataframe = pd.DataFrame(initial_data)

        original_dataframe.to_excel(excel_writer=self.file_name, index=False)

        return original_dataframe

    def read_excel(self):
        read_excel_dataframe = pd.read_excel(io=self.file_name)
        print(read_excel_dataframe)


    def update_excel(self):

        original_dataframe = pd.read_excel(io=self.file_name)

        original_dataframe['Email'] = [ 1, 2, 3, 4]

        original_dataframe["Experience"] = original_dataframe["Age"] - 16

        original_dataframe.loc[original_dataframe["Age"] <= 18, "Experience"] = None

        print(original_dataframe)

        original_dataframe.to_excel(excel_writer=self.file_name, index=False)

        return original_dataframe

    def delete_excel(self, df):

        df = df.drop(columns=['Email'])
        df = df.drop(index=2)
        df = df.reset_index()
        print(df)




excel_crud_obj = ExcelCRUD(file_name='students_data.xlsx')

# excel_crud_obj.create_excel()
# excel_crud_obj.read_excel()
updated_dataframe = excel_crud_obj.update_excel()
excel_crud_obj.delete_excel(updated_dataframe)
