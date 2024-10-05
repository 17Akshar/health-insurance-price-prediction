from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json


app = FastAPI()


@app.get('/')
def hello_world():
    return "Hello,World"


model = pickle.load(open('prediction.pxl','rb'))
class model_input(BaseModel):

    age : int
    sex : int
    weight : int
    height : int
    children : int
    smoker : int
    region : int


@app.post("/predict")
def insurance_prediction(input_params: model_input):
    input_data = input_params.json()
    input_dict = json.loads(input_data)

    age = input_dict['age']
    sex = input_dict['sex']
    bmi = input_dict['weight']/input_dict['height']
    children = input_dict['children']
    smoker = input_dict['smoker']
    region = input_dict['region']

    input_list = [age,sex,bmi,children,smoker,region]

    prediction_result = model.predict([input_list])


        # Return the result in a JSON format
    return {"prediction": prediction_result[0]}




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
