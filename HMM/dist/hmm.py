"""A student-friendly Hidden Markov Model for the hidden two-coin experiment.

===============================================================================
1. WHAT IS HIDDEN, AND WHAT IS OBSERVED?
===============================================================================

Imagine that a dealer is behind a curtain. The dealer chooses one of two coins,
flips it, and tells us only whether the result was heads or tails.

Hidden states (the thing we want to discover):

    S = {Fair, Biased}

Observations (the thing we can actually hear):

    V = {H, T}

For example, in ``H H H T`` we observe four results, but we do not know which
coin produced each result. An HMM describes this situation with three groups of
probabilities. Textbooks often write the complete model as lambda = (pi, A, B).


===============================================================================
2. THE THREE PARTS OF THIS HMM: pi, A, AND B
===============================================================================

(a) INITIAL-STATE DISTRIBUTION pi
---------------------------------

``pi[i]`` is the probability that the FIRST hidden state is state i.

With an equal starting chance:

                Fair    Biased
    pi =       [0.50,     0.50]

In this file, pi is named ``initial_distribution`` and is stored in
``model["initial"]``.


(b) TRANSITION MATRIX A
-----------------------

``A[i][j]`` is the probability of moving FROM hidden state i TO hidden state j.
Rows mean "from" and columns mean "to".

With an 80% chance of keeping either coin:

                         TO
                    Fair    Biased
              Fair  0.80     0.20
    FROM    Biased  0.20     0.80

The rows sum to 1 because, after each toss, the next coin must be either Fair or
Biased. In this file, A is named ``transition_matrix`` and is stored in
``model["transitions"]``. For example:

    model["transitions"]["Fair"]["Biased"] == 0.20

means "there is a 20% chance of switching from Fair to Biased."


(c) EMISSION MATRIX B
---------------------

``B[j][symbol]`` is the probability that hidden state j EMITS an observation.
Rows are hidden coins and columns are visible results.

                         OBSERVATION
                       H          T
              Fair    0.50       0.50
            Biased    0.85       0.15

The rows also sum to 1 because a toss must be either heads or tails. In this
file, B is named ``emission_matrix`` and is stored in ``model["emissions"]``.
For example:

    model["emissions"]["Biased"]["H"] == 0.85

means "the biased coin has an 85% chance of emitting heads."


===============================================================================
3. WHAT QUESTION DOES VITERBI ANSWER?
===============================================================================

Given the observations O = (o_1, o_2, ..., o_T), Viterbi finds ONE complete
hidden-state sequence Q* that has the largest joint probability:

    Q* = argmax_Q P(Q, O | model)

It does not try every complete path. At each time and for each possible current
state, it remembers only:

1. the score of the best path that ends in that state; and
2. a back-pointer to the previous state on that best path.

This is dynamic programming: two partial paths that end in the same state have
exactly the same future choices, so only the better partial path can ever win.

The function ``viterbi`` below is divided into the four standard textbook phases:

    Step 1: Initialization
    Step 2: Recursion
    Step 3: Termination
    Step 4: Backtracking

The equations and their matching Python statements are written beside each step.


===============================================================================
4. HOW THIS FILE IS USED
===============================================================================

``analyse`` builds the model and runs the algorithms. This Python file is the
fully explained teaching and test version. The offline browser uses a matching
translation in ``hmm.js``, which returns the same path, trellis, and probability
values without downloading a Python runtime. Keeping both versions lets students
study Python while letting the web lesson work anywhere.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any


STATES = ("Fair", "Biased")
OBSERVATIONS = ("H", "T")


def _check_probability(name: str, value: float) -> float:
    """Return a probability as a float, or explain why it is invalid."""

    # Change slider-like input into a decimal number such as 0.85.
    number = float(value)
    # A probability is a piece of one whole pie. This lesson keeps every slice
    # strictly between an empty pie (0) and the complete pie (1).
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be greater than 0 and less than 1.")
    return number


def _clean_observations(sequence: str | Iterable[str]) -> list[str]:
    """Convert a friendly sequence such as ``H-H T`` into ``['H', 'H', 'T']``."""

    if isinstance(sequence, str):
        # These marks are only visual spacers; remove them like gaps between
        # letter tiles while keeping the H/T tiles themselves.
        separators = {" ", "\t", "\n", "\r", ",", "-", "–", "—", ">"}
        observations = [character.upper() for character in sequence if character not in separators]
    else:
        observations = [str(item).upper() for item in sequence]

    if not observations:
        raise ValueError("The observation sequence must contain at least one toss.")
    if len(observations) > 60:
        raise ValueError("The observation sequence may contain at most 60 tosses.")

    invalid = [item for item in observations if item not in OBSERVATIONS]
    if invalid:
        raise ValueError("Every observation must be H (heads) or T (tails).")
    return observations


def make_model(
    *,
    fair_heads: float = 0.50,
    biased_heads: float = 0.85,
    fair_stay: float = 0.80,
    biased_stay: float = 0.80,
    initial_fair: float = 0.50,
) -> dict[str, Any]:
    """Build pi, transition matrix A, and emission matrix B.

    Each row must total 1. For example, if there is an 80% chance of keeping the
    fair coin, there is automatically a 20% chance of switching to the biased coin.

    Dictionary rows are used instead of numeric row indexes so a learner can read
    ``transition_matrix["Fair"]["Biased"]`` without memorising that Fair is row 0.
    """

    fair_heads = _check_probability("fair_heads", fair_heads)
    biased_heads = _check_probability("biased_heads", biased_heads)
    fair_stay = _check_probability("fair_stay", fair_stay)
    biased_stay = _check_probability("biased_stay", biased_stay)
    initial_fair = _check_probability("initial_fair", initial_fair)

    # INITIAL DISTRIBUTION pi
    # -----------------------
    # pi[state] = P(the first hidden state is state)
    initial_distribution = {
        "Fair": initial_fair,
        "Biased": 1.0 - initial_fair,
    }

    # TRANSITION MATRIX A
    # -------------------
    # transition_matrix[from_state][to_state] = P(to_state | from_state)
    #
    #                              TO
    #                         Fair       Biased
    # FROM  Fair          fair_stay   1 - fair_stay
    #       Biased  1-biased_stay       biased_stay
    transition_matrix = {
        "Fair": {
            "Fair": fair_stay,
            "Biased": 1.0 - fair_stay,
        },
        "Biased": {
            "Fair": 1.0 - biased_stay,
            "Biased": biased_stay,
        },
    }

    # EMISSION MATRIX B
    # -----------------
    # emission_matrix[state][observation] = P(observation | state)
    #
    #                         OBSERVATION
    #                         H              T
    #       Fair          fair_heads    1 - fair_heads
    #       Biased      biased_heads  1 - biased_heads
    emission_matrix = {
        "Fair": {
            "H": fair_heads,
            "T": 1.0 - fair_heads,
        },
        "Biased": {
            "H": biased_heads,
            "T": 1.0 - biased_heads,
        },
    }

    # These keys keep the mathematical parts separate and easy to inspect.
    return {
        "initial": initial_distribution,       # pi
        "transitions": transition_matrix,      # A
        "emissions": emission_matrix,          # B
    }


def viterbi(observations: list[str], model: dict[str, Any]) -> dict[str, Any]:
    """Use the Viterbi algorithm to find the most likely hidden-state sequence.

    TEXTBOOK SYMBOLS USED IN THE COMMENTS

    * ``o_t``       = observation at time t, such as H
    * ``i``         = a possible previous state
    * ``j``         = a possible current state
    * ``A[i][j]``   = transition probability from i to j
    * ``B[j][o_t]`` = probability that state j emits observation o_t
    * ``delta_t(j)``= score of the best partial path ending in j at time t
    * ``psi_t(j)``  = previous state on that best partial path

    The four phases are:

    1. INITIALIZATION

       delta_1(j) = pi[j] * B[j][o_1]

       Score starting in each state and emitting the first observation.

    2. RECURSION for t = 2, ..., T

       delta_t(j) = max_i(delta_(t-1)(i) * A[i][j]) * B[j][o_t]
       psi_t(j)   = argmax_i(delta_(t-1)(i) * A[i][j])

       For each current state j, compare all ways of entering it. Save the best
       score in delta and the winning previous state in psi.

    3. TERMINATION

       q_T* = argmax_j delta_T(j)

       Choose the state with the best score at the final time.

    4. BACKTRACKING

       q_t* = psi_(t+1)(q_(t+1)*)

       Follow the saved psi pointers backward to reconstruct the complete path.

    Why logarithms? A path probability multiplies many numbers smaller than one.
    A long product can become too tiny for a computer to store accurately.
    Logarithms turn multiplication into addition:

        log(x * y) = log(x) + log(y)

    Therefore, this implementation stores log(delta) instead of delta. Taking the
    maximum is unchanged because the logarithm is an increasing function.
    """

    # ``delta`` stores the best log-score for every [time][current_state].
    # It is the Viterbi probability table, often drawn as a trellis in textbooks.
    delta: list[dict[str, float]] = []

    # ``psi`` stores the winning previous state for every trellis cell. These
    # pointers are what make it possible to reconstruct the winning path later.
    psi: list[dict[str, str | None]] = []

    # ``trellis`` is extra teaching information used by the web interface. The
    # Viterbi calculation itself needs only delta and psi.
    trellis: list[dict[str, dict[str, float | str | None]]] = []

    # =======================================================================
    # VITERBI STEP 1 OF 4: INITIALIZATION
    # =======================================================================
    # Formula: delta_1(j) = pi[j] * B[j][o_1]
    #
    # With the default first observation H:
    #
    #   Fair:   0.50 starting chance * 0.50 chance of H = 0.250
    #   Biased: 0.50 starting chance * 0.85 chance of H = 0.425
    #
    # Biased has the better first score, but we keep BOTH scores because either
    # state may become part of the best complete path after later observations.
    first_observation = observations[0]
    delta_at_first_time: dict[str, float] = {}
    psi_at_first_time: dict[str, str | None] = {}
    first_details: dict[str, dict[str, float | str | None]] = {}

    for current_state in STATES:
        # pi[j]: probability of starting in current_state.
        initial_probability = model["initial"][current_state]

        # B[j][o_1]: probability that current_state emits the first result.
        emission_probability = model["emissions"][current_state][first_observation]

        # In normal probability space this is multiplication. In log space it
        # becomes addition: log(delta_1(j)) = log(pi[j]) + log(B[j][o_1]).
        initial_log_score = math.log(initial_probability) + math.log(emission_probability)

        delta_at_first_time[current_state] = initial_log_score

        # There is no previous state at t=1, so the first psi pointer is None.
        psi_at_first_time[current_state] = None

        first_details[current_state] = {
            "best_previous": None,
            "initial_probability": initial_probability,
            "transition_probability": None,
            "emission_probability": emission_probability,
            "path_probability": math.exp(initial_log_score),
        }

    delta.append(delta_at_first_time)
    psi.append(psi_at_first_time)
    trellis.append(first_details)

    # =======================================================================
    # VITERBI STEP 2 OF 4: RECURSION
    # =======================================================================
    # Move left-to-right through observations 2, 3, ..., T. For every current
    # state j, compare the route from Fair and the route from Biased.
    #
    # Probability-space formula:
    #
    #   delta_t(j) = max_i(delta_(t-1)(i) * A[i][j]) * B[j][o_t]
    #
    # Log-space formula used by the code:
    #
    #   log_delta_t(j) = max_i(log_delta_(t-1)(i) + log(A[i][j]))
    #                    + log(B[j][o_t])
    for time_index in range(1, len(observations)):
        observation = observations[time_index]
        delta_at_this_time: dict[str, float] = {}
        psi_at_this_time: dict[str, str] = {}
        details_now: dict[str, dict[str, float | str | None]] = {}

        for current_state in STATES:
            # First calculate the score of every route entering current_state.
            # This is delta_(t-1)(i) * A[i][j], expressed in log space.
            incoming_log_scores = {
                previous_state: (
                    delta[time_index - 1][previous_state]
                    + math.log(model["transitions"][previous_state][current_state])
                )
                for previous_state in STATES
            }

            # psi_t(j) = argmax_i(...)
            # Save WHICH previous state supplied the strongest incoming route.
            best_previous_state = max(
                STATES,
                key=lambda previous_state: incoming_log_scores[previous_state],
            )

            # B[j][o_t]: after entering current_state, that state must emit the
            # observation heard at this time.
            emission_probability = model["emissions"][current_state][observation]

            # delta_t(j): take the winning incoming route, then include the
            # current emission. Multiplication again becomes addition in logs.
            best_log_score = (
                incoming_log_scores[best_previous_state]
                + math.log(emission_probability)
            )

            delta_at_this_time[current_state] = best_log_score
            psi_at_this_time[current_state] = best_previous_state

            details_now[current_state] = {
                "best_previous": best_previous_state,
                "initial_probability": None,
                "transition_probability": model["transitions"][best_previous_state][current_state],
                "emission_probability": emission_probability,
                "path_probability": math.exp(best_log_score),
            }

        delta.append(delta_at_this_time)
        psi.append(psi_at_this_time)
        trellis.append(details_now)

    # =======================================================================
    # VITERBI STEP 3 OF 4: TERMINATION
    # =======================================================================
    # Formula: q_T* = argmax_j delta_T(j)
    # The last trellis column contains one best route ending in Fair and one best
    # route ending in Biased. The larger of those two scores wins overall.
    final_state = max(STATES, key=lambda state: delta[-1][state])
    best_log_probability = delta[-1][final_state]

    # =======================================================================
    # VITERBI STEP 4 OF 4: BACKTRACKING
    # =======================================================================
    # Formula: q_t* = psi_(t+1)(q_(t+1)*)
    # Start with the winning final state. Each psi pointer tells us which state
    # came immediately before it on the winning route.
    reversed_path = [final_state]
    for time_index in range(len(observations) - 1, 0, -1):
        current_state = reversed_path[-1]
        previous_state = psi[time_index][current_state]
        assert previous_state is not None  # Only the very first pointer is None.
        reversed_path.append(previous_state)

    # We collected states from T back to 1, so reverse them into chronological order.
    most_likely_path = list(reversed(reversed_path))

    return {
        "path": most_likely_path,
        "path_probability": math.exp(best_log_probability),
        "path_log_probability": best_log_probability,
        "trellis": trellis,
    }


def forward_backward(observations: list[str], model: dict[str, Any]) -> dict[str, Any]:
    """Find the probability of each state at each step using all observations.

    IMPORTANT: this function is not the Viterbi algorithm. Viterbi returns one
    best complete path. Forward-backward instead returns a probability distribution
    over states at every individual time. The web app uses these probabilities for
    the Fair/Biased bars shown when a learner selects a toss.

    The forward pass asks, "How likely is everything heard up to here?" The
    backward pass asks, "How likely is everything still to come?" Multiplying
    both answers gives a posterior probability for the hidden coin at that step.

    Each forward column is scaled to total 1. The saved scale factors let us
    calculate the sequence likelihood while keeping the numbers numerically safe.
    """

    # The forward table carries evidence left-to-right, like passing a note down
    # a line of children. Each column stores Fair and Biased shares.
    forward: list[dict[str, float]] = []
    # Scale factors keep tiny multiplied probabilities large enough to store.
    scales: list[float] = []

    # First toss: starting chance × chance this coin emits the heard result.
    first = {
        state: model["initial"][state] * model["emissions"][state][observations[0]]
        for state in STATES
    }
    # Add both route weights, then divide by the total so the two shares make 1.
    first_scale = sum(first.values())
    forward.append({state: first[state] / first_scale for state in STATES})
    scales.append(first_scale)

    # Move through every later observation from left to right.
    for time_index in range(1, len(observations)):
        observation = observations[time_index]
        column: dict[str, float] = {}

        for state_now in STATES:
            # Two roads can arrive here: from Fair or from Biased. Add both road
            # weights because forward-backward keeps every possible path.
            arriving_probability = sum(
                forward[time_index - 1][state_before]
                * model["transitions"][state_before][state_now]
                for state_before in STATES
            )
            # The arriving route must also produce today's heard H or T.
            column[state_now] = arriving_probability * model["emissions"][state_now][observation]

        # Normalize this column to two friendly shares that add to 1.
        scale = sum(column.values())
        forward.append({state: column[state] / scale for state in STATES})
        scales.append(scale)

    # The backward table carries future evidence right-to-left. At the final
    # toss, there is no future evidence, so both states begin with neutral 1.
    backward: list[dict[str, float]] = [{state: 0.0 for state in STATES} for _ in observations]
    backward[-1] = {state: 1.0 for state in STATES}

    # Walk backward from the second-last toss toward the first.
    for time_index in range(len(observations) - 2, -1, -1):
        next_observation = observations[time_index + 1]
        for state_now in STATES:
            # Add both possible next roads: transition × next emission × the
            # evidence still farther ahead. Divide by the saved scale for safety.
            backward[time_index][state_now] = sum(
                model["transitions"][state_now][state_next]
                * model["emissions"][state_next][next_observation]
                * backward[time_index + 1][state_next]
                for state_next in STATES
            ) / scales[time_index + 1]

    # Glue the left-looking and right-looking evidence together at every toss.
    posterior: list[dict[str, float]] = []
    for time_index in range(len(observations)):
        # Multiplying forward × backward scores how well each coin explains the
        # entire sequence while standing at this one time.
        unnormalised = {
            state: forward[time_index][state] * backward[time_index][state]
            for state in STATES
        }
        # Divide by both scores together so Fair + Biased = 100%.
        total = sum(unnormalised.values())
        posterior.append({state: unnormalised[state] / total for state in STATES})

    # Multiplying all saved scales gives the probability of hearing the sequence
    # across every possible hidden path. Logs turn that product into a safe sum.
    sequence_log_probability = sum(math.log(scale) for scale in scales)
    return {
        "posterior": posterior,
        "sequence_probability": math.exp(sequence_log_probability),
        "sequence_log_probability": sequence_log_probability,
    }


def analyse(
    sequence: str | Iterable[str],
    *,
    fair_heads: float = 0.50,
    biased_heads: float = 0.85,
    fair_stay: float = 0.80,
    biased_stay: float = 0.80,
    initial_fair: float = 0.50,
) -> dict[str, Any]:
    """Build the HMM, decode its best path, and calculate state probabilities.

    Reading order for a new student:

    1. ``make_model`` creates pi, transition matrix A, and emission matrix B.
    2. ``viterbi`` finds the one most likely hidden coin sequence.
    3. ``forward_backward`` supplies the separate per-step probability bars.
    """

    observations = _clean_observations(sequence)
    model = make_model(
        fair_heads=fair_heads,
        biased_heads=biased_heads,
        fair_stay=fair_stay,
        biased_stay=biased_stay,
        initial_fair=initial_fair,
    )
    decoded = viterbi(observations, model)
    probabilities = forward_backward(observations, model)

    return {
        "observations": observations,
        "model": model,
        **decoded,
        **probabilities,
    }


def analyse_json(config_json: str) -> str:
    """A tiny, dependable doorway between JavaScript and Python.

    JSON contains only plain strings, numbers, lists, and dictionaries. Using it
    here avoids surprising conversion rules at the WebAssembly language boundary.
    """

    config = json.loads(config_json)
    result = analyse(
        config["sequence"],
        fair_heads=config.get("fair_heads", 0.50),
        biased_heads=config.get("biased_heads", 0.85),
        fair_stay=config.get("fair_stay", 0.80),
        biased_stay=config.get("biased_stay", 0.80),
        initial_fair=config.get("initial_fair", 0.50),
    )
    return json.dumps(result, separators=(",", ":"))
