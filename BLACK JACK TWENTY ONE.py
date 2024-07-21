import tkinter as tk
from tkinter import messagebox, ttk
import random
from collections import Counter
import itertools
import math

class MCTSNode:
    def __init__(self, player_total, dealer_visible_cards, true_count, parent=None):
        self.player_total = player_total
        self.dealer_visible_cards = dealer_visible_cards
        self.true_count = true_count
        self.parent = parent
        self.children = {'hit': None, 'stand': None}
        self.visits = 0
        self.wins = 0.0

    def ucb_score(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_weight * exploration

    def is_terminal(self):
        return self.player_total >= 21

class MCTS:
    def __init__(self, deck):
        self.deck = deck

    def run(self, root_node, num_simulations):
        for _ in range(num_simulations):
            node = self.select(root_node)
            if not node.is_terminal():
                node = self.expand(node)
            outcome = self.simulate(node)
            self.backpropagate(node, outcome)
        return self.best_action(root_node)

    def select(self, node):
        while not node.is_terminal():
            if node.children['hit'] is None or node.children['stand'] is None:
                return node
            node = max(node.children.values(), key=lambda n: n.ucb_score())
        return node

    def expand(self, node):
        if node.children['hit'] is None:
            new_total = node.player_total + random.choice(self.deck)
            new_node = MCTSNode(new_total, node.dealer_visible_cards, node.true_count, parent=node)
            node.children['hit'] = new_node
            return new_node
        elif node.children['stand'] is None:
            new_node = MCTSNode(node.player_total, node.dealer_visible_cards, node.true_count, parent=node)
            node.children['stand'] = new_node
            return new_node
        return node

    def simulate(self, node):
        player_total = node.player_total
        dealer_total = sum(node.dealer_visible_cards)

        # Player's turn
        while player_total < 21:
            if random.random() < 0.5:  # 50% chance to hit
                player_total += random.choice(self.deck)
            else:
                break

        # Dealer's turn
        while dealer_total < 17:
            dealer_total += random.choice(self.deck)

        if player_total > 21:
            return 'Lose'  # Player busts
        elif dealer_total > 21 or player_total > dealer_total:
            return 'Win'  # Player wins
        elif player_total < dealer_total:
            return 'Lose'  # Player loses
        else:
            return 'Push'  # Push

    def backpropagate(self, node, outcome):
        while node is not None:
            node.visits += 1
            node.wins += self.outcome_to_value(outcome)  # Convert outcome to numeric value
            node = node.parent

    def outcome_to_value(self, outcome):
        if outcome == 'Win':
            return 1.0
        elif outcome == 'Lose':
            return 0.0
        elif outcome == 'Push':
            return 0.5
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

    def best_action(self, node):
        hit_node = node.children['hit']
        stand_node = node.children['stand']
        if hit_node.visits > stand_node.visits:
            return 'hit', hit_node.wins / hit_node.visits
        else:
            return 'stand', stand_node.wins / stand_node.visits

class BlackjackAdvisor:
    def __init__(self, master):
        self.master = master
        master.title("BJ")
        master.geometry("800x600")
        master.configure(bg="#f0f0f0")

        self.initialize_deck()

        # Create main frame
        main_frame = tk.Frame(master, bg="#f0f0f0")
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Input frame
        input_frame = tk.LabelFrame(main_frame, text="Input", bg="#f0f0f0", font=("Arial", 12, "bold"))
        input_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(input_frame, text="Your cards (comma-separated):", bg="#f0f0f0", font=("Arial", 10)).grid(row=0, column=0, sticky="e", pady=5)
        self.your_cards_entry = tk.Entry(input_frame, font=("Arial", 10))
        self.your_cards_entry.grid(row=0, column=1, pady=5, padx=5, sticky="we")

        tk.Label(input_frame, text="Dealer's upcard:", bg="#f0f0f0", font=("Arial", 10)).grid(row=1, column=0, sticky="e", pady=5)
        self.dealer_upcard_entry = tk.Entry(input_frame, font=("Arial", 10))
        self.dealer_upcard_entry.grid(row=1, column=1, pady=5, padx=5, sticky="we")

        tk.Label(input_frame, text="Risk Tolerance:", bg="#f0f0f0", font=("Arial", 10)).grid(row=2, column=0, sticky="e", pady=5)
        self.risk_tolerance = ttk.Combobox(input_frame, values=["Low", "Medium", "High"], font=("Arial", 10))
        self.risk_tolerance.set("Medium")
        self.risk_tolerance.grid(row=2, column=1, pady=5, padx=5, sticky="we")

        tk.Label(input_frame, text="Card Counting System:", bg="#f0f0f0", font=("Arial", 10)).grid(row=3, column=0, sticky="e", pady=5)
        self.counting_system = ttk.Combobox(input_frame, values=["Hi-Lo", "KO", "Omega II", "CCSP"], font=("Arial", 10))
        self.counting_system.set("Hi-Lo")
        self.counting_system.grid(row=3, column=1, pady=5, padx=5, sticky="we")

        tk.Label(input_frame, text="Algorithm:", bg="#f0f0f0", font=("Arial", 10)).grid(row=4, column=0, sticky="e", pady=5)
        self.algorithm = ttk.Combobox(input_frame, values=["Monte Carlo", "SB Theory", "MCTS", "SEMCTS"], font=("Arial", 10))
        self.algorithm.set("Monte Carlo")
        self.algorithm.grid(row=4, column=1, pady=5, padx=5, sticky="we")

        calculate_button = tk.Button(input_frame, text="Calculate", command=self.calculate, bg="#00e5ff", fg="white", font=("Arial", 10, "bold"))
        calculate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Result frame
        result_frame = tk.LabelFrame(main_frame, text="Result", bg="#f0f0f0", font=("Arial", 12, "bold"))
        result_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, height=20, width=70, font=("Arial", 10))
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the result text
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

    def initialize_deck(self):
        self.deck = list(range(1, 12))  # Assuming cards from 1 to 11 (Ace to Jack)
        self.count = 0
        self.used_cards = []

    def calculate(self):
        self.initialize_deck()
        your_cards_str = self.your_cards_entry.get()
        dealer_upcard_str = self.dealer_upcard_entry.get()

        # Validate input format
        if not all(card_str.replace(" ", "").isdigit() for card_str in your_cards_str.split(",") + dealer_upcard_str.split(",")):
            messagebox.showwarning("Invalid Input", "Please enter card values as numbers separated by commas.")
            return

        your_cards = [int(card) for card in your_cards_str.split(",")]
        # Parse dealer's visible cards
        dealer_visible_cards = [int(card.strip()) for card in self.dealer_upcard_entry.get().split(',')]
        dealer_upcard = dealer_visible_cards[0]  # The first visible card is the upcard
        risk_tolerance = self.risk_tolerance.get()
        counting_system = self.counting_system.get()
        algorithm = self.algorithm.get()

        # Validate cards
        all_cards = your_cards + dealer_visible_cards
        if len(set(all_cards)) != len(all_cards) or any(card not in self.deck for card in all_cards):
            messagebox.showwarning("Invalid Input", "Invalid card combination. Please enter unique cards from 1 to 11.")
            return

        # Update used cards and deck
        for card in all_cards:
            self.deck.remove(card)
            self.used_cards.append(card)

        your_total = sum(your_cards)
        true_count = self.count / (len(self.deck) / 11)

        if algorithm == "MCTS":
            mcts = MCTS(self.deck)
            root_node = MCTSNode(your_total, dealer_visible_cards, true_count)
            action, win_rate = mcts.run(root_node, num_simulations=100000)
            recommendation = f"{action.upper()} (MCTS win rate: {win_rate:.2%})"
            result = self.format_results(your_total, dealer_visible_cards, {'win_rate': win_rate}, risk_tolerance, counting_system, algorithm, recommendation)
        elif algorithm == "SB Theory":
            sb_results = self.scenario_bruteforce(your_total, dealer_visible_cards)
            recommendation = self.get_sb_recommendation(sb_results)
            result = self.format_results(your_total, dealer_visible_cards, sb_results, risk_tolerance, counting_system, algorithm, recommendation)
        elif algorithm == "SEMCTS":
            semcts_results = self.run_semcts(your_total, dealer_visible_cards, true_count, risk_tolerance)
            decision, hit_score, stand_score = self.semcts_decision(your_total, dealer_visible_cards, true_count, risk_tolerance)
            recommendation = f"{decision} (Hit score: {hit_score:.4f}, Stand score: {stand_score:.4f})"
            result = self.format_results(your_total, dealer_visible_cards, semcts_results, risk_tolerance, counting_system, algorithm, recommendation)
        else:  # Monte Carlo
            simulation_results = self.monte_carlo_simulation(your_total, dealer_visible_cards)
            self.update_count(all_cards, counting_system)
            recommendation = self.get_recommendation(
                simulation_results['hit_ev'],
                simulation_results['stand_ev'],
                your_total,
                dealer_visible_cards,
                true_count,
                risk_tolerance
            )
            result = self.format_results(your_total, dealer_visible_cards, simulation_results, risk_tolerance, counting_system, algorithm, recommendation)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

    def monte_carlo_simulation(self, your_total, dealer_visible_cards):
        results = {'hit': Counter(), 'stand': Counter()}
        simulations = 1000000

        for _ in range(simulations):
            simulation_deck = self.deck.copy()
            random.shuffle(simulation_deck)

            if not simulation_deck:
                break

            dealer_total = sum(dealer_visible_cards)
            while dealer_total < 17 and simulation_deck:
                dealer_total += simulation_deck.pop()

            hit_total = your_total
            if hit_total < 21 and simulation_deck:
                hit_card = simulation_deck.pop()
                hit_total += hit_card

            results['hit'][self.determine_outcome(hit_total, dealer_total)] += 1
            results['stand'][self.determine_outcome(your_total, dealer_total)] += 1

        hit_total = sum(results['hit'].values())
        stand_total = sum(results['stand'].values())

        hit_win_prob = (results['hit']['Win'] + results['hit']['Push'] / 2) / hit_total if hit_total > 0 else 0
        stand_win_prob = (results['stand']['Win'] + results['stand']['Push'] / 2) / stand_total if stand_total > 0 else 0

        hit_ev = hit_win_prob - (1 - hit_win_prob)
        stand_ev = stand_win_prob - (1 - stand_win_prob)

        return {
            'hit': results['hit'],
            'stand': results['stand'],
            'hit_ev': hit_ev,
            'stand_ev': stand_ev,
            'hit_total': hit_total,
            'stand_total': stand_total
        }

    def scenario_bruteforce(self, your_total, dealer_visible_cards):
        results = {'hit': Counter(), 'stand': Counter()}
        step_by_step = []
        future_considerations = {'hit': 0, 'stand': 0}

        dealer_total = sum(dealer_visible_cards)
        dealer_upcard = dealer_visible_cards[0]

        step_by_step.append(f"Starting SB Theory simulation:\n")
        step_by_step.append(f"Your total: {your_total}")
        step_by_step.append(f"Dealer's visible cards: {dealer_visible_cards}\n")

        for dealer_hidden in self.deck:
            current_dealer_total = dealer_total + dealer_hidden
            step_by_step.append(f"Scenario: Dealer's hidden card is {dealer_hidden}")
            step_by_step.append(f"Dealer's current total: {current_dealer_total}")

            # Dealer draws cards until reaching at least 17
            while current_dealer_total < 17:
                additional_card = random.choice(self.deck)
                current_dealer_total += additional_card
                step_by_step.append(f"Dealer draws: {additional_card}, New total: {current_dealer_total}")

            # Stand scenario
            stand_outcome = self.determine_outcome(your_total, current_dealer_total)
            results['stand'][stand_outcome] += 1
            step_by_step.append(f"  If you STAND: {stand_outcome}")

            # Hit scenarios
            step_by_step.append("  If you HIT:")
            for hit_card in self.deck:
                new_total = your_total + hit_card
                hit_outcome = self.determine_outcome(new_total, current_dealer_total)
                results['hit'][hit_outcome] += 1
                step_by_step.append(f"   Draw {hit_card}: New total {new_total} - Outcome: {hit_outcome}")

                # Future considerations
                if hit_outcome == 'Win':
                    future_considerations['hit'] += 1
                elif hit_outcome == 'Lose':
                    future_considerations['hit'] -= 1

            if stand_outcome == 'Win':
                future_considerations['stand'] += 1
            elif stand_outcome == 'Lose':
                future_considerations['stand'] -= 1

            step_by_step.append("")  # Empty line for readability

        hit_total = sum(results['hit'].values())
        stand_total = sum(results['stand'].values())

        hit_win_prob = (results['hit']['Win'] + results['hit']['Push'] / 2) / hit_total if hit_total > 0 else 0
        stand_win_prob = (results['stand']['Win'] + results['stand']['Push'] / 2) / stand_total if stand_total > 0 else 0

        step_by_step.append(f"Final probabilities:")
        step_by_step.append(f"Hit win probability: {hit_win_prob:.2%}")
        step_by_step.append(f"Stand win probability: {stand_win_prob:.2%}")
        step_by_step.append(f"Future considerations (Hit): {future_considerations['hit']}")
        step_by_step.append(f"Future considerations (Stand): {future_considerations['stand']}")

        return {
            'hit': results['hit'],
            'stand': results['stand'],
            'hit_win_prob': hit_win_prob,
            'stand_win_prob': stand_win_prob,
            'hit_total': hit_total,
            'stand_total': stand_total,
            'step_by_step': step_by_step,
            'future_considerations': future_considerations
        }

    def run_semcts(self, your_total, dealer_visible_cards, true_count, risk_tolerance):
        sb_results = self.scenario_bruteforce(your_total, dealer_visible_cards)
        
        # Step 2: MCTS Initialization
        mcts = MCTS(self.deck)
        root_node = MCTSNode(your_total, dealer_visible_cards, true_count)

        # Step 3: Guided MCTS Simulations
        num_simulations = 100000
        for _ in range(num_simulations):
            node = mcts.select(root_node)
            if not node.is_terminal():
                node = mcts.expand(node)
            outcome = self.guided_simulate(node, sb_results)
            mcts.backpropagate(node, outcome)

        # Step 4: Combined Decision
        mcts_hit_prob = root_node.children['hit'].wins / root_node.children['hit'].visits if root_node.children['hit'] else 0
        mcts_stand_prob = root_node.children['stand'].wins / root_node.children['stand'].visits if root_node.children['stand'] else 0

        sb_hit_prob = sb_results['hit_win_prob']
        sb_stand_prob = sb_results['stand_win_prob']

        risk_factor = self.get_risk_factor(risk_tolerance)
        mcts_weight = 0.6 * risk_factor
        sb_weight = 0.4 / risk_factor

        combined_hit_prob = (mcts_hit_prob * mcts_weight + sb_hit_prob * sb_weight) / (mcts_weight + sb_weight)
        combined_stand_prob = (mcts_stand_prob * mcts_weight + sb_stand_prob * sb_weight) / (mcts_weight + sb_weight)

        return {
            'combined_hit_prob': combined_hit_prob,
            'combined_stand_prob': combined_stand_prob,
            'mcts_hit_prob': mcts_hit_prob,
            'mcts_stand_prob': mcts_stand_prob,
                        'sb_hit_prob': sb_hit_prob,
            'sb_stand_prob': sb_stand_prob,
            'your_total': your_total,
            'dealer_visible_cards': dealer_visible_cards
        }

    def guided_simulate(self, node, sb_results):
        player_total = node.player_total
        dealer_visible_cards = node.dealer_visible_cards
        dealer_total = sum(dealer_visible_cards)
        
        # Use scenario analysis to guide the simulation
        hit_prob = sb_results['hit_win_prob']

        # Player's turn
        while player_total < 21:
            if random.random() < hit_prob:  # Probability to hit based on scenario analysis
                player_total += self.weighted_card_draw(node.true_count)
            else:
                break

        # Dealer's turn
        while dealer_total < 17:
            dealer_total += self.weighted_card_draw(node.true_count)

        return self.determine_outcome(player_total, dealer_total)

    def weighted_card_draw(self, true_count):
        weights = [1] * len(self.deck)
        for i, card in enumerate(self.deck):
            if card in [2, 3, 4, 5, 6]:
                weights[i] += 0.1 * true_count
            elif card in [10, 11]:
                weights[i] -= 0.1 * true_count
        return random.choices(self.deck, weights=weights)[0]

    def get_semcts_recommendation(self, results):
        if results['combined_hit_prob'] > results['combined_stand_prob']:
            return f"HIT (Combined probability: {results['combined_hit_prob']:.2%})"
        else:
            return f"STAND (Combined probability: {results['combined_stand_prob']:.2%})"

    def determine_outcome(self, player_total, dealer_total):
        if player_total > 21:
            return 'Lose'
        elif dealer_total > 21:
            return 'Win'
        elif player_total > dealer_total:
            return 'Win'
        elif player_total < dealer_total:
            return 'Lose'
        else:
            return 'Push'

    def format_results(self, your_total, dealer_visible_cards, results, risk_tolerance, counting_system, algorithm, recommendation):
        output = "Final probabilities:\n"
        
        if algorithm == "MCTS":
            output += f"MCTS Win Rate: {results['win_rate']:.2%}\n"
        elif algorithm == "SB Theory":
            output += f"Hit win probability: {results['hit_win_prob']:.2%}\n"
            output += f"Stand win probability: {results['stand_win_prob']:.2%}\n"
            if 'future_considerations' in results:
                output += f"Future considerations (Hit): {results['future_considerations']['hit']:.2f}\n"
                output += f"Future considerations (Stand): {results['future_considerations']['stand']:.2f}\n"
        elif algorithm == "SEMCTS":
            output += f"Combined Hit Probability: {results['combined_hit_prob']:.2%}\n"
            output += f"Combined Stand Probability: {results['combined_stand_prob']:.2%}\n"
            output += f"MCTS Hit Probability: {results['mcts_hit_prob']:.2%}\n"
            output += f"MCTS Stand Probability: {results['mcts_stand_prob']:.2%}\n"
            output += f"SB Theory Hit Probability: {results['sb_hit_prob']:.2%}\n"
            output += f"SB Theory Stand Probability: {results['sb_stand_prob']:.2%}\n"
        else:  # Monte Carlo
            hit_total = results['hit_total']
            stand_total = results['stand_total']
            output += f"Hit win probability: {(results['hit']['Win'] + results['hit']['Push'] / 2) / hit_total:.2%}\n"
            output += f"Stand win probability: {(results['stand']['Win'] + results['stand']['Push'] / 2) / stand_total:.2%}\n"

        output += f"Recommendation: {recommendation}\n\n"
        
        output += f"Your total: {your_total}\n"
        output += f"Dealer's visible cards: {dealer_visible_cards}\n"
        output += f"Cards remaining in deck: {len(self.deck)}\n"
        output += f"Risk Tolerance: {risk_tolerance}\n"
        output += f"Card Counting System: {counting_system}\n"
        output += f"Algorithm: {algorithm}\n\n"

        true_count = self.count / (len(self.deck) / 11)
        dynamic_weights = self.get_dynamic_weights(true_count)

        output += f"Running count: {self.count}\n"
        output += f"True count: {true_count:.2f}\n"
        output += f"Count level: {dynamic_weights['count_level']}\n"
        output += f"Dynamic hit weight: {dynamic_weights['hit']:.2f}\n"
        output += f"Dynamic stand weight: {dynamic_weights['stand']:.2f}\n\n"

        if algorithm == "SB Theory":
            output += "Step-by-step SB Theory simulation:\n"
            output += "\n".join(results['step_by_step'])
        elif algorithm == "SEMCTS":
            if 'sb_results' in results and 'step_by_step' in results['sb_results']:
                output += "SB Theory Step-by-step simulation:\n"
                output += "\n".join(results['sb_results']['step_by_step'])
        elif algorithm == "Monte Carlo":
            output += f"Expected value of hitting: {results['hit_ev']:.2f}\n"
            output += f"Expected value of standing: {results['stand_ev']:.2f}\n"
            output += f"Weighted expected value of hitting: {results['hit_ev'] * dynamic_weights['hit'] * self.get_risk_factor(risk_tolerance):.2f}\n"

        return output

    def get_recommendation(self, hit_ev, stand_ev, your_total, dealer_visible_cards, true_count, risk_tolerance):
        risk_factor = self.get_risk_factor(risk_tolerance)
        dynamic_weights = self.get_dynamic_weights(true_count)

        weighted_hit_ev = hit_ev * dynamic_weights['hit'] * risk_factor
        weighted_stand_ev = stand_ev * dynamic_weights['stand'] / risk_factor

        if weighted_hit_ev > weighted_stand_ev:
            return "HIT"
        else:
            return "STAND"

    def get_dynamic_weights(self, true_count):
        count_level = self.get_count_level(true_count)

        weight_adjustments = {
            "Very Low": (-0.2, 0.2),
            "Low": (-0.1, 0.1),
            "Slightly Low": (-0.05, 0.05),
            "Neutral": (0, 0),
            "Slightly High": (0.05, -0.05),
            "High": (0.1, -0.1),
            "Very High": (0.2, -0.2)
        }

        hit_adjustment, stand_adjustment = weight_adjustments[count_level]

        hit_weight = 1.0 + hit_adjustment
        stand_weight = 1.0 + stand_adjustment

        return {
            'hit': hit_weight,
            'stand': stand_weight,
            'count_level': count_level
        }

    def get_sb_recommendation(self, sb_results):
        hit_win_prob = sb_results['hit_win_prob']
        stand_win_prob = sb_results['stand_win_prob']
        risk_tolerance = self.risk_tolerance.get()
        risk_factor = self.get_risk_factor(risk_tolerance)
        true_count = self.count / (len(self.deck) / 11)
        dynamic_weights = self.get_dynamic_weights(true_count)

        weighted_hit_prob = hit_win_prob * dynamic_weights['hit'] * risk_factor
        weighted_stand_prob = stand_win_prob * dynamic_weights['stand'] / risk_factor

        sb_results['weighted_hit_prob'] = weighted_hit_prob
        sb_results['weighted_stand_prob'] = weighted_stand_prob
        sb_results['count_level'] = dynamic_weights['count_level']

        future_hit = sb_results['future_considerations']['hit']
        future_stand = sb_results['future_considerations']['stand']

        hit_score = weighted_hit_prob + future_hit * 0.1
        stand_score = weighted_stand_prob + future_stand * 0.1

        if abs(hit_score - stand_score) < 0.05:
            if risk_tolerance == "Low":
                return "STAND (Conservative play due to close probabilities)"
            elif risk_tolerance == "High":
                return "HIT (Aggressive play due to close probabilities)"

        if dynamic_weights['count_level'] in ["High", "Very High"] and stand_score > 0.45:
            return "STAND (Favorable count for standing)"
        elif dynamic_weights['count_level'] in ["Low", "Very Low"] and hit_score > 0.45:
            return "HIT (Unfavorable count for standing)"

        return "HIT" if hit_score > stand_score else "STAND"

    def update_count(self, cards, counting_system):
        for card in cards:
            if counting_system == "Hi-Lo":
                if card in [2, 3, 4, 5, 6]:
                    self.count += 1
                elif card in [10, 11]:  # Assuming 11 represents Ace here
                    self.count -= 1
            elif counting_system == "KO":
                if card in [2, 3, 4, 5, 6, 7]:
                    self.count += 1
                elif card in [10, 11]:
                    self.count -= 1
            elif counting_system == "Omega II":
              if card in [2, 3, 7]:
                self.count += 1
              elif card in [4, 5, 6]:
                self.count += 2
              elif card in [9]:
                self.count -= 1
              elif card in [10, 11]:
                self.count -= 2
            elif counting_system == "CCSP":
              if card in [2, 3, 4, 5, 6]:
                self.count += 1
              elif card in [10, 11]:
                self.count -= 1

    def get_risk_factor(self, risk_tolerance):
        if risk_tolerance == "Low":
            return 0.8
        elif risk_tolerance == "High":
            return 1.2
        else:  # Medium
            return 1.0

    def calculate_future_considerations(self, current_total, action):
        remaining_cards = Counter(self.deck)
        total_cards = sum(remaining_cards.values())
        
        if action == 'hit':
            possible_totals = {}
            for card, count in remaining_cards.items():
                new_total = current_total + card
                if new_total <= 21:
                    possible_totals[new_total] = count / total_cards
        else:  # stand
            possible_totals = {current_total: 1}
        
        weighted_future_score = sum((21 - total) * prob for total, prob in possible_totals.items())
        return weighted_future_score

    def get_count_level(self, true_count):
        if true_count <= -3:
            return "Very Low"
        elif -3 < true_count <= -1:
            return "Low"
        elif -1 < true_count < 0:
            return "Slightly Low"
        elif true_count == 0:
            return "Neutral"
        elif 0 < true_count < 1:
            return "Slightly High"
        elif 1 <= true_count < 3:
            return "High"
        else:
            return "Very High"

    def semcts_decision(self, your_total, dealer_visible_cards, true_count, risk_tolerance):
        # Step 1: Run SEMCTS
        semcts_results = self.run_semcts(your_total, dealer_visible_cards, true_count, risk_tolerance)
        
        # Step 2: Calculate decision scores
        hit_score = self.calculate_decision_score('hit', semcts_results, true_count, risk_tolerance)
        stand_score = self.calculate_decision_score('stand', semcts_results, true_count, risk_tolerance)
        
        # Step 3: Make decision
        if hit_score > stand_score:
            return 'HIT', hit_score, stand_score
        else:
            return 'STAND', hit_score, stand_score

    def calculate_decision_score(self, action, semcts_results, true_count, risk_tolerance):
        # Base probabilities
        mcts_prob = semcts_results[f'mcts_{action}_prob']
        sb_prob = semcts_results[f'sb_{action}_prob']
        combined_prob = semcts_results[f'combined_{action}_prob']
        
        # Weights
        w_mcts = 0.4
        w_sb = 0.3
        w_combined = 0.3
        
        # Risk factor (0.8 for low, 1.0 for medium, 1.2 for high)
        risk_factor = self.get_risk_factor(risk_tolerance)
        
        # Count adjustment
        count_adjustment = self.get_count_adjustment(true_count, action)
        
        # Calculate base score
        base_score = (w_mcts * mcts_prob + w_sb * sb_prob + w_combined * combined_prob) * risk_factor
        
        # Apply count adjustment
        adjusted_score = base_score + count_adjustment
        
        # Apply action-specific adjustments
        if action == 'hit':
            # Encourage hitting on lower totals
            total_adjustment = max(0, (21 - semcts_results['your_total']) / 21)
            adjusted_score *= (1 + total_adjustment)
        else:  # stand
            # Encourage standing on higher totals
            total_adjustment = max(0, (semcts_results['your_total'] - 11) / 10)
            adjusted_score *= (1 + total_adjustment)
        
        # Normalize score to be between 0 and 1
        final_score = math.tanh(adjusted_score)
        
        return final_score

    def get_count_adjustment(self, true_count, action):
        if action == 'hit':
            return -0.05 * true_count  # Discourage hitting as count increases
        else:  # stand
            return 0.05 * true_count  # Encourage standing as count increases

root = tk.Tk()
advisor = BlackjackAdvisor(root)
root.mainloop()
    