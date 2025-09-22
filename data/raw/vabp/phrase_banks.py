# --------------------------------------------------
# |            Individual Phrase Banks             |
# --------------------------------------------------
root_consider_phrases = [
    "Ok let's consider starting with {move_description}.",
    "How about we analyze the line starting with {move_description}?",
    "Let's consider playing {move_description}.",
    "Ok, what if we played {move_description}?",
    "Let's think through {move_description}.",
    "What if we played {move_description}?",
    "We could try playing {move_description}.",
    "We could play {move_description}.",
    "What if we start with {move_description}?",
]

write_off_root_phrases = [
    "No, I don't like this direction.",
    "This isn't the best line, let's consider alternatives.",
    "Actually, this isn't a great idea.",
    "Nevermind - let's consider something else.",
    "Wait, I don't like this line let's think about something else.",
    "Hold that, this seems like a bad direction. Let's consider other lines.",
    "Actually - this doesn't look promising. Let's think through other moves.",
]

our_move_first_child_phrases = [
    "We could then respond with {move_description}.",
    "Ok, then we could play {move_description}.",
    "We could follow with {move_description}.",
    "What if we then played {move_description}?",
    "Then we might play {move_description}.",
    "We could then move {move_description}.",
    "We could consider following with {move_description}."
]

our_move_sibling_phrases = [
    "Ok we could instead play {move_description}.",
    "Instead we could move {move_description}.",
    "Alternatively, we could do {move_description}.",
    "Another possible move for us could be {move_description}.",
    "Or we could try playing {move_description}.",
    "What if instead we moved {move_description}?",
    "We could also play {move_description}.",
    "We could also try {move_description}.",
    "Another option would be {move_description}."
]

opponent_move_first_child_phrases = [
    "A strong response from them would be {move_description}.",
    "They could respond with {move_description}.",
    "They might play {move_description} in response.",
    "A smart move from them could be {move_description}.",
    "From here they'll likely consider {move_description}."
]

opponent_move_sibling_phrases = [
    "They could also answer with {move_description}.",
    "They might also consider {move_description}.",
    "They could also consider {move_description}.",
    "As an alternative they could play {move_description}.",
    "They could instead move {move_description}.",
    "They might also think about {move_description}.",
    "They may also think about moving {move_description}.",
]

us_best_move_leaf_phrases = [
    "Of all of these, we should play {move_description}.", 
    "Out of those options, the best would be {move_description}.", 
    "We would choose {move_description} out of all the options.", 
    "I think of these we would choose {move_description}.", 
    "The best of which would be {move_description}.", 
]

us_best_move_branch_phrases = [
    "Given how the lines would play out for {uci_list}, we should play {move_description}.",
    "Of our possible moves ({uci_list}) and after thinking through how the opponent would respond, our best play would be {move_description}.",
    "Given the likely outcome of each of our possible moves ({uci_list}), we should choose {move_description}.",
    "After simulating optimal play from our opponent, of the moves {uci_list} we should play {move_description}.",
    "Following our analysis of how each move would play out from {uci_list}, the best move for us would be {move_description}."
]

opponent_best_move_leaf_phrases = [
    "I would expect them to play {move_description} as that is their better move.",
    "If they play optimally, they would choose {move_description}.",
    "Of these, their best move would be {move_description}.",
    "Their optimal choice should be {move_description}.",
    "Their best move of these would be {move_description} -- I expect they would play that.",
    "They should choose {move_description} of these since it is stronger.",
]

opponent_best_move_branch_phrases = [
    "If the opponent played optimally, of the moves {uci_list} they would likely choose {move_description} as this gives them the best position.",
    "Knowing how each line would likely play out, from their moves ({uci_list}) the opponent would likely choose {move_description}.",
    "Of their available moves - {uci_list} - the opponent's best move would be {move_description}.",
    "Of their moves ({uci_list}), they should optimally play {move_description}.",
    "The opponent should choose optimally from their moves ({uci_list}) by playing {move_description}.",
]

us_prune_branch_phrases = [
    "No, this branch doesn't seem right.",
    "Nevermind - let's consider alternatives.",
    "Actually, that isn't what we should play.",
    "No, maybe let's consider another move."
]

# This should be called VERY rarely (if at all)
opponent_prune_branch_phrases = [
    "Actually they probably won't play that.",
    "On second thought they wouldn't choose this branch.",
]

board_valuation_phrases = [
    ("This resulting board is overwhelmingly in our favor {board_value}.", 675),
    ("The position after this is completely dominant for our side {board_value}.", 675),
    ("This line gives us a position that feels like we'll win {board_value}.", 675),
    ("Looking at this position, this feels like we're on a path to victory {board_value}.", 660),
    ("After this line, we're in a very favorable position to win {board_value}.", 630),
    ("The resulting position gives us a massive advantage {board_value}.", 600),
    ("This sequence leaves us with an overwhelming edge {board_value}.", 600),
    ("The position after these moves is definitely favorable for us {board_value}.", 570),
    ("This leaves us with a board that is clearly to our advantage {board_value}.", 525),
    ("Looking at this position, we have a strong advantage {board_value}.", 450),
    ("The resulting position gives us a clear edge {board_value}.", 420),
    ("This sequence leaves us with fair chance of winning {board_value}.", 375),
    ("After these moves, we maintain a solid advantage {board_value}.", 330),
    ("This line leaves us with a nice advantage {board_value}.", 270),
    ("The position after this seems to favor our side {board_value}.", 255),
    ("Looking at this, we appear to be in a better position than our opponent {board_value}.", 225),
    ("This position feels somewhat better for our side {board_value}.", 180),
    ("The resulting position seems to give us a small edge {board_value}.", 150),
    ("After these moves, we seem to have a slightly better position than them {board_value}.", 105),
    ("This line seems to leave us marginally ahead {board_value}.", 105),
    ("The position feels like it could be slightly in our favor {board_value}.", 105),
    ("Looking at this, we might have a tiny advantage {board_value}.", 75),
    ("The resulting position appears to give us a minimal edge {board_value}.", 60),
    ("This line leaves things mostly even, perhaps slightly ahead for us {board_value}.", 45),
    ("The position feels roughly balanced, maybe a touch in our favor {board_value}.", 30),
    ("After these moves, things look approximately equal {board_value}.", 0),
    ("This results in what seems like a neutral board position {board_value}.", 0),
    ("The position after this appears essentially balanced {board_value}.", 0),
    ("Looking at this, neither side seems to have an edge {board_value}.", 0),
    ("This line leaves us in an equal position compared to the opponent {board_value}.", 0),
    ("This position seems slightly better for the opponent {board_value}.", -45),
    ("Looking at this, our position seems marginally behind {board_value}.", -75),
    ("The resulting position appears to favor them slightly {board_value}.", -105),
    ("This line leaves us a bit worse off {board_value}.", -135),
    ("After these moves, the opponent seems to have an edge {board_value}.", -180),
    ("The position has the opponent ahead {board_value}.", -225),
    ("This line leaves us at a clear disadvantage {board_value}.", -270),
    ("The resulting position gives the opponent a strong advantage {board_value}.", -315),
    ("After these moves, the board clearly has us behind {board_value}.", -375),
    ("Looking at this position, we're clearly struggling {board_value}.", -375),
    ("The position after this puts us in a difficult spot {board_value}.", -420),
    ("This sequence leaves us with very poor winning chances {board_value}.", -450),
    ("This line has us in a very troubling position {board_value}.", -450),
    ("After these moves, we're facing a serious uphill battle {board_value}.", -495),
    ("The resulting position will require us to really make a come back {board_value}.", -495),
    ("Looking at this, our position has us playing from very far behind {board_value}.", -525),
    ("Our position after these moves is very unfavorable for winning {board_value}.", -570),
    ("The resulting position has us on the verge of losing {board_value}.", -600),
    ("After this sequence, we're left in an extremely difficult position {board_value}.", -630),
    ("This board position is nearly hopeless for us to come back from {board_value}.", -675),
    ("Looking at this position, we're facing low odds of coming back from this {board_value}.", -675),
    ("The resulting position leaves us with very little chance to come back {board_value}.", -675)
]

final_statement_phrases = [
    "After analyzing these lines, our best move would be to choose {best_move}.",
    "Given how these various moves would play out, we should pick {best_move}.",
    "Our best move given how these would play out against optimal opponent play would be {best_move}.",
    "Of these moves we should choose to play {best_move}.",
    "Out of all of these options we should choose to play {best_move}.",
    "Given how each line would evolve we should pick {best_move} as this puts us in the best position."
]

# --------------------------------------------------
# |          Deprecated / Semi-Deprecated          |
# --------------------------------------------------
excellent_move_phrases = [
    "That looks like a really strong move{move_value}.",
    "Hm this seems like a very good move for us{move_value}.",
    "Looks like it could be an excellent move{move_value}.",
    "May be a brilliant move{move_value}!",
    "I think that's a super strong move{move_value}."
]

good_move_phrases = [
    "Looks to be a good move{move_value}.", 
    "Seems like a strong move{move_value}.",
    "I think this is a positive line to consider{move_value}.",
    "This feels like a good direction{move_value}.",
    "Just thinking out loud, seems like a good move{move_value}.",
    "Good move - I like this{move_value}."
]

bad_move_phrases = [
    "Looks like a poor line{move_value}.",
    "Doesn't seem like a positive direction{move_value}.",
    "Not a great move{move_value}.",
    "This feels like a suboptimal direction{move_value}.",
    "Seems like a bleh move{move_value}."
]

blunder_phrases = [
    "Bleh, may be a blunder{move_value}.",
    "Nope this seems like a very bad direction{move_value}.",
    "That seems like a very bad move{move_value}.",
    "Hm, seems like a big mistake{move_value}.",
    "Oh no we'd definitely be in trouble here{move_value}."
]


# Combined phrase bank dict for easier passing through
phrase_banks = {
    "root_consider_phrases": root_consider_phrases,
    "write_off_root_phrases": write_off_root_phrases,
    "our_move_first_child_phrases": our_move_first_child_phrases,
    "our_move_sibling_phrases": our_move_sibling_phrases,
    "opponent_move_first_child_phrases": opponent_move_first_child_phrases,
    "opponent_move_sibling_phrases": opponent_move_sibling_phrases,
    "us_best_move_leaf_phrases": us_best_move_leaf_phrases,
    "us_best_move_branch_phrases": us_best_move_branch_phrases,
    "opponent_best_move_leaf_phrases": opponent_best_move_leaf_phrases,
    "opponent_best_move_branch_phrases": opponent_best_move_branch_phrases,
    "us_prune_branch_phrases": us_prune_branch_phrases,
    "board_valuation_phrases": board_valuation_phrases,
    "opponent_prune_branch_phrases": opponent_prune_branch_phrases,
    "final_statement_phrases": final_statement_phrases,
    "excellent_move_phrases": excellent_move_phrases,
    "good_move_phrases": good_move_phrases,
    "bad_move_phrases": bad_move_phrases,
    "blunder_phrases": blunder_phrases,
}


# --------------------------------------------------
# |     Phrase Bank for Generating Train Data      |
# --------------------------------------------------
initial_think_phrase = [
    "Ok let's think through this position.",
    "Interesting, let's consider some moves here.",
    "Got it, let's think about some lines we could play.",
    "Great let's think through this position.",
    "Ok let's consider some moves we could make.",
    "Let's think through this.",
]