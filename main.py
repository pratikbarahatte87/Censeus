import pickle
import numpy as np
import streamlit as st
from sklearn import tree
import pandas as pd

st.set_page_config(page_title="Predict Risk Category",layout="wide")  # title shown in the browser tab

for model_number in ["1","2","3","4"]:
    model_inputs = f"inputs_{model_number}"
    if model_number not in st.session_state:
        st.session_state[model_number] = None
    if model_inputs not in st.session_state:
        st.session_state[model_inputs] = None


if "state_counter" not in st.session_state:
    st.session_state["state_counter"] = 0

def count_inputs(df):
    sum_yes = list()
    sum_no = list()
    for i in range(len(df)):
        row = list(df.iloc[i])
        sum_yes.append(row.count('Yes'))
        sum_no.append(row.count('No'))
    return sum_yes, sum_no

def convert_df(df):
   return df.to_csv().encode('utf-8')

def validate_generic_inputs(m_inputs):
    """
    validate the user input

    If any input is undefined then print the warning and returns False
    In case every input is defined return True

    Args:
        inputs: list of all the inputs by user

    Return:
        validation result
        True if validation successful else False
    """
    # take indices of all undefined values
    indices = [i for i, x in enumerate(m_inputs[:]) if x == 5]
    # list of input labels
    input_nm = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
    ]

    # extract names of all undefined values
    ll = [input_nm[i] for i in indices]
    # prompt user to define input if undefined
    if len(ll) > 0:
        st.markdown(
            '<p style="color: rgb(255, 75, 75)">ERROR: You have not defined the following inputs from above<br><p>&emsp;&emsp;&emsp;{}\n<p style="color: rgb(255, 75, 75)">Please define the inputs to run the model</p>'.format(
                "<br>&emsp;&emsp;&emsp; ".join(ll)
            ),
            unsafe_allow_html=True,
        )

    # return the validation result in form of True or False
    return True if len(ll) == 0 else False

def validate_summary_inputs():

    if None in st.session_state.values():
        st.markdown("# ERROR:")
        for model in ["1","2","3","4"]:
            output = st.session_state[model]
            if output is None:
                st.markdown(f"#### No inputs selected for Model {model}")
    else:
        return True

## function to make the prediction POST request to IBM watson studio
def make_request(input_data, model):

    prediction = model.predict(
        list([input_data]),
        list([[0]]),
    )
    return prediction

## this function convert input of race according to model requirement
def e_race(x):
    """
    this function convert input of race according to model requirement
    """
    pass

def generic_model_page(model_number):

    ### set page properties
    # CONVERT TO COLUMNS
    col1, col2 = st.columns(2)
    state_key = str(st.session_state["state_counter"])
    # column one
    with col1:
        # streamlit radio input
        A = st.radio("A", ("undefined", "Yes", "No"), index=1, key=state_key)
        C = st.radio("C", ("undefined", "Yes", "No"), index=1, key=state_key)
        E = st.radio("E", ("undefined", "Yes", "No"), index=2, key=state_key)
        G = st.radio("G", ("undefined", "Yes", "No"), index=1, key=state_key)
        I = st.radio("I", ("undefined", "Yes", "No"), index=2, key=state_key)
        K = st.radio("K", ("undefined", "Yes", "No"), index=1, key=state_key)
        M = st.radio("M", ("undefined", "Yes", "No"), index=1, key=state_key)
        O = st.radio("O", ("undefined", "Yes", "No"), index=1, key=state_key)
        Q = st.radio("Q", ("undefined", "Yes", "No"), index=2, key=state_key)
        S = st.radio("S", ("undefined", "Yes", "No"), index=1, key=state_key)
        U = st.radio("U", ("undefined", "Yes", "No"), index=2, key=state_key)
        W = st.radio("W", ("undefined", "Yes", "No"), index=1, key=state_key)


    # column two
    with col2:
        B = st.radio("B", ("undefined", "Yes", "No"), index=2, key=state_key)
        D = st.radio("D", ("undefined", "Yes", "No"), index=1, key=state_key)
        F = st.radio("F", ("undefined", "Yes", "No"), index=2, key=state_key)
        H = st.radio("H", ("undefined", "Yes", "No"), index=1, key=state_key)
        J = st.radio("J", ("undefined", "Yes", "No"), index=2, key=state_key)
        L = st.radio("L", ("undefined", "Yes", "No"), index=1, key=state_key)
        N = st.radio("N", ("undefined", "Yes", "No"), index=2, key=state_key)
        P = st.radio("P", ("undefined", "Yes", "No"), index=1, key=state_key)
        R = st.radio("R", ("undefined", "Yes", "No"), index=1, key=state_key)
        T = st.radio("T", ("undefined", "Yes", "No"), index=2, key=state_key)
        V = st.radio("V", ("undefined", "Yes", "No"), index=2, key=state_key)

    ### when button clicked
    # create st.button to make API request
    predict_btn = st.button("predict")
    loaded_model = pickle.load(open("model.pkl", "rb"))

    # when clicked
    if predict_btn:
        # collect inputs from radio button

        inputs = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W]
        input_nms = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
        ]
        # # convert input values according to model's requirements
        m_inputs = [0 if i == "No" else 1 if i == "Yes" else 5 for i in inputs[:]]
        # take indices of all 0 values
        indices_0 = [i for i, x in enumerate(m_inputs) if x == 0]
        # take indices of all 1 values
        indices_1 = [i for i, x in enumerate(m_inputs) if x == 1]

        # # validate inputs
        is_validated = validate_generic_inputs(m_inputs)

        # if validated
        if is_validated:
            # make an TODO
            result = make_request(m_inputs, loaded_model)
            st.session_state[model_number] = result[0]
            model_inputs = f"inputs_{model_number}"
            st.session_state[model_inputs] = m_inputs
            st.markdown("## Input Summary ")
            st.markdown(
                '<h4 style="color: rgb(255, 75, 75)">List of inputs as Yes: </h4>',
                unsafe_allow_html=True,
            )
            if len(indices_1) > 0:
                st.markdown(
                    f'{"&emsp;" * 3}{", ".join(str(v) for v in [input_nms[i] for i in indices_1])} ',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f'{"&emsp;" * 3}None')

            st.markdown(
                '<h4 style="color: rgb(255, 75, 75)">List of inputs as No: </h4>',
                unsafe_allow_html=True,
            )
            if len(indices_0) > 0:
                st.markdown(
                    f'{"&emsp;" * 3}{", ".join(str(v) for v in [input_nms[i] for i in indices_0])} ',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f'{"&emsp;" * 3}None')

            if None in st.session_state.values():
                for model in ["1","2","3","4"]:
                    output = st.session_state[model]
                    if output is None:
                        st.markdown(f"#### Please select inputs for Model {model}")
                    else:
                        st.markdown(f"#### Inputs for Model {model} accepted!")
            else:
                st.markdown(f"#### Please go to Model 5 page for consensus results")

def model_five_summary_page():

    if validate_summary_inputs():

        input_nms = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
        ]

        summary_columns = ['Model']
        summary_columns.extend(input_nms)

        input_summary = pd.DataFrame(index=range(4), columns=summary_columns)
        for i in range(4):
            model_inputs = st.session_state[f"inputs_{i + 1}"]
            transformed_inputs = ["Yes" if input == 1 else "No" for input in model_inputs]
            final_inputs = [f"{i + 1}"]
            final_inputs.extend(transformed_inputs)
            input_summary.iloc[i] = final_inputs


        input_summary['Yes Count'], input_summary['No Count'] = count_inputs(input_summary)
        input_summary['Prediction'] = [
                                      st.session_state["1"],
                                      st.session_state["2"],
                                      st.session_state["3"],
                                      st.session_state["4"]
                                      ]

        mean_result = input_summary['Prediction'].mean()
        input_summary['Mean Prediction'] = mean_result



        st.markdown("# Modelwise User Inputs:")
        st.dataframe(input_summary)

        csv = convert_df(input_summary)

        st.download_button(
           "Download This Table",
           csv,
           "modelwise_user_inputs.csv",
           "text/csv",
           key='download-csv'
        )

        input_model = pickle.load(open("model.pkl", "rb"))
        summary_model = pickle.load(open("model5.pkl", "rb"))
        summary_inputs = np.array([
                          st.session_state["1"],
                          st.session_state["2"],
                          st.session_state["3"],
                          st.session_state["4"]
                          ]).reshape(1, -1)

        summary_mean = sum(summary_inputs[0])/len(summary_inputs[0])
        prediction = summary_model.predict(summary_inputs)[0]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"# Model 5 Result: {prediction}")
        with col2:
            st.markdown(f"# Mean: {mean_result}")
        # st.markdown(f"#### Model 1 Prediction: {st.session_state['1']}")
        # st.markdown(f"#### Model 2 Prediction: {st.session_state['2']}")
        # st.markdown(f"#### Model 3 Prediction: {st.session_state['3']}")
        # st.markdown(f"#### Model 4 Prediction: {st.session_state['4']}")
        # st.markdown(f"#### Mean Model Prediction: {summary_mean}")
        # st.markdown(f"## Model 5 Consensus Prediction: {prediction}")

        input_model_importances = pd.DataFrame({'Feature':input_nms,'Importance':input_model.feature_importances_})

        summary_importances = pd.DataFrame({'Feature':['Model 1','Model 2','Model 3','Model 4'],'Importance':summary_model.feature_importances_})

        with col1:
            st.markdown(f"##### Model 1 Feature Importances")
            st.dataframe(input_model_importances, height=1000)
            st.markdown(f"##### Model 3 Feature Importances")
            st.dataframe(input_model_importances, height=1000)

        with col2:
            st.markdown(f"##### Model 2 Feature Importances")
            st.dataframe(input_model_importances, height=1000)
            st.markdown(f"##### Model 4 Feature Importances")
            st.dataframe(input_model_importances, height=1000)

        st.markdown(f"##### Model 5 Feature Importances")
        st.dataframe(summary_importances, width=500)

        # fig = plt.figure(figsize=(75,60))
        # _ = tree.plot_tree(input_model,
        #                    feature_names=input_nms,
        #                    class_names=["0","1","2"],
        #                    filled=True,
        #                    fontsize=12)
        # fig.savefig("input_decision_tree.png")
        #
        # fig = plt.figure(figsize=(75,60))
        # _ = tree.plot_tree(summary_model,
        #                    feature_names=['Model 1','Model 2','Model 3','Model 4'],
        #                    class_names=["0","1","2"],
        #                    filled=True,
        #                    fontsize=12)
        # fig.savefig("summary_decision_tree.png")





page = st.selectbox("Choose your page", ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5 (Consensus)", "Model Architectures"])

if page == "Model 1":
    st.session_state["state_counter"] = "a"
    generic_model_page("1")
if page == "Model 2":
    st.session_state["state_counter"] = "b"
    generic_model_page("2")
if page == "Model 3":
    st.session_state["state_counter"] = "c"
    generic_model_page("3")
if page == "Model 4":
    st.session_state["state_counter"] = "d"
    generic_model_page("4")
if page == "Model 5 (Consensus)":
    model_five_summary_page()
if page == "Model Architectures":
    st.markdown(f"# Models 1-4 Architecture")
    st.image("input_decision_tree.png")
    st.markdown(f"# Model 5 Architecture")
    st.image("summary_decision_tree.png")
