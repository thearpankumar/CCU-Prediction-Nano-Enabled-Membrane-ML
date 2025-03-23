
---

# Nano-Material Adsorption Prediction for CCU

This project uses an XGBoost machine learning model to predict the adsorption capacity (`nads`) of nano-materials for Carbon Capture and Utilization (CCU). The goal is to identify optimal nano-material configurations that efficiently capture CO₂ and H₂O molecules, supporting the development of advanced membranes, sorbents, and catalysts as described in the research paper *"Nano-enabled Membranes, Sorbents, and Catalysts for Addressing the Challenges of Carbon Capture and Utilization (CCU)"*. The model is trained on simulation data (e.g., `is2r_train_optimized.csv`), saved for reuse, and can predict adsorption capacity for new nano-material designs.

## Project Overview
- **Objective**: Predict `nads` (number of adsorbed molecules) to rank nano-material configurations by their CO₂ capture efficiency.
- **Model**: XGBoost regression with parameters optimized for accuracy (RMSE evaluation).
- **Dataset**: Based on `is2r_train_optimized.csv`, a simulation dataset of nano-material properties.
- **Output**: Predicted `nads` values, identifying top-performing configurations for CCU applications.

## Installation
### Prerequisites
- Python 3.7+
- Libraries: Install via `pip`:
  ```bash
  pip install pandas numpy xgboost scikit-learn matplotlib
  ```

### Setup
1. Clone or download this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Place your training data (`is2r_train_optimized.csv`) in the project directory, or update the file path in `train_xgboost.py`.
3. Ensure new input data follows the required format (see [Input Data Format](#input-data-format)).

### 2. Making Predictions
Use the saved model to predict `nads` for new data:
```bash
python predict_xgboost.py
```
- **What It Does**:
  - Loads `xgboost_nads_model.json`.
  - Predicts `nads` for your input data (e.g., `your_new_data.csv`).
  - Saves results to `predictions_output.csv`.
- **Steps**:
  1. Prepare your input data (see [Input Data Format](#input-data-format)).
  2. Update the file path in `predict_xgboost.py` (e.g., `"your_new_data.csv"`).
  3. Run the script.
- **Output Example**:
  ```
  Model loaded successfully from 'xgboost_nads_model.json'
  Predictions for the first 5 rows:
     nco2  nh2o  y_init  natoms  defective  ...  predicted_nads
  0     1     2  0.6126      82          1  ...        3.0318
  1     1     0 -0.4206      95          0  ...        2.9876
  Results saved to 'predictions_output.csv'
  ```



### Required Features
| Feature               | Description                            | Data Type    | Example       |
|-----------------------|----------------------------------------|--------------|---------------|
| `nco2`               | Number of CO₂ molecules exposed        | Integer/Float| 1             |
| `nh2o`               | Number of H₂O molecules exposed        | Integer/Float| 2             |
| `y_init`             | Initial energy (eV)                    | Float        | 0.612635      |
| `natoms`             | Number of atoms in material            | Integer      | 82            |
| `defective`          | Has defects? (True/False or 1/0)      | Boolean/Int  | True          |
| `cell_mean`          | Average cell size (Å)                  | Float        | 5.643653      |
| `cell_std`           | Std dev of cell size (Å)               | Float        | 4.755599      |
| `pos_relaxed_mean`   | Mean relaxed position (Å)              | Float        | 8.48211       |
| `pos_relaxed_std`    | Std dev of relaxed position (Å)        | Float        | 4.277549      |
| `atomic_numbers_mean`| Mean atomic number                     | Float        | 13.573171     |
| `atomic_numbers_std` | Std dev of atomic numbers              | Float        | 12.533885     |

### Example CSV
```csv
nco2,nh2o,y_init,natoms,defective,cell_mean,cell_std,pos_relaxed_mean,pos_relaxed_std,atomic_numbers_mean,atomic_numbers_std
1,2,0.612635,82,True,5.643653,4.755599,8.48211,4.277549,13.573171,12.533885
1,0,-0.420629,95,False,5.373723,5.147071,7.654403,3.760834,6.905263,12.555620
```

- **Notes**: 
  - Use the same units (e.g., Å for distances, eV for energy) as the training data.
  - `defective` can be `True`/`False` or `1`/`0`; the script converts it to 0/1.

## Project Goals
- Identify high-performing nano-materials for CCU by predicting `nads`.
- Support research into efficient CO₂ capture, aligning with nano-enabled sorbent/membrane advancements.
- Provide a reusable model for testing new material designs.

## Future Improvements
- Add CO₂-specific selectivity metrics (current `nads` includes H₂O).
- Integrate with neural networks for deeper pattern analysis.
- Expand to predict stability (`raw_y`) alongside capacity.

## Contributing
Feel free to fork this repo, submit issues, or suggest enhancements via pull requests!

---

### Notes
- **Customization**: Adjust the repo URL, license, or file paths as needed for your setup (e.g., if hosted on GitHub).
- **Data**: I didn’t include `is2r_train_optimized.csv` since it’s your proprietary file—users should have their own version.
- **Tone**: It’s practical and user-friendly, aimed at researchers or developers in CCU.

Want to tweak anything (e.g., add more details, change the structure)? Let me know! You can save this as `README.md` in your project folder.