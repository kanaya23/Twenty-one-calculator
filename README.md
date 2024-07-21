# 21 Advisor Tool Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Input Parameters](#input-parameters)
5. [Algorithms](#algorithms)
6. [Output Interpretation](#output-interpretation)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Introduction

The 21 Advisor Tool is not a sophisticated application designed to assist players in making optimal decisions during a game of 21. It utilizes various algorithms and strategies to provide recommendations based on the current game state, card counting, and risk tolerance. this is created for the 21 in Resident Evil.

Note: This tool doesn't include the trump cards

## Installation

1. Ensure you have Python 3.7 or higher installed on your system.
2. Clone the repository or download the source code.
3. Install required dependencies:
   ```
   pip install tkinter
   ```

## Usage

To run the 21 Advisor Tool:

1. Navigate to the directory containing the script.
2. Run the following command:
   ```
   python "BLACK JACK TWENTY ONE.py"
   ```
3. The graphical user interface (GUI) will appear, allowing you to input game parameters and receive advice.

## Input Parameters

- **Your cards**: Enter your current hand total.
- **Dealer's visible cards**: Enter the dealer's visible cards, separated by commas.
- **Risk Tolerance**: Choose between Low, Medium, or High.
- **Card Counting System**: Select from Hi-Lo, KO, Omega II, or CCSP.
- **Algorithm**: Choose between Monte Carlo, SB Theory, MCTS, or SEMCTS.

## Algorithms

1. **Monte Carlo**: Simulates multiple game outcomes to estimate probabilities.
2. **SB (Scenario Bruteforce)**: Analyzes all possible scenarios to determine the best action.
3. **MCTS (Monte Carlo Tree Search)**: Uses a tree search algorithm to explore decision space efficiently.
4. **SEMCTS (Scenario-Enhanced Monte Carlo Tree Search)**: Combines scenario analysis with MCTS for improved accuracy.

## Output Interpretation

The tool provides the following information:

- Win probabilities for hitting and standing
- Recommended action (Hit or Stand)
- Current game state (your total, dealer's cards, etc.)
- Card counting information (running count, true count)
- Algorithm-specific details

## Advanced Features

### Card Counting
The tool supports various card counting systems and adjusts its recommendations based on the current count.

### Dynamic Weighting
Recommendations are dynamically weighted based on the true count and risk tolerance.

### Future Considerations
The tool analyzes potential future scenarios to provide more accurate long-term advice.

## Troubleshooting

If you encounter any issues:

1. Ensure all input fields are filled correctly.
2. Check that you've selected valid options for all dropdown menus.
3. Verify that your Python environment meets the requirements.

note: I will not be updating the code, fixing bugs, or providing any further support. However, I am open to pull requests if you would like to contribute improvements or fixes, also the code is a fever dream and a nightmare, but it works.
