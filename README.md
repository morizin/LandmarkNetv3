# LandmarkNetv3

## Running the Script

To run the `src/main.py` script, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/morizin/LandmarkNetv3.git
    cd LandmarkNetv3
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and extract the dataset**:
    ```bash
    ./get_dataset.sh
    ```

4. **Run the script**:
    ```bash
    python src/main.py --data_dir <path_to_dataset> --output_dir <path_to_output>
    ```

### Arguments

- `--data_dir`: Path to the dataset directory.
- `--output_dir`: Path to the output directory where results will be saved.

### Example

```bash
python src/main.py --data_dir ./essentials/loftr-repo --output_dir ./output
```
