import ffai
import ffai.web.api as api
import ffai.web.server as server
from ffai.ai.registry import make_bot

import grodbot
import scripted_bot_example
import test_random
import test_proc

#other_agent = ffai.Agent("Player 1", human=True)
other_agent = make_bot("terriblebot")
#other_agent = make_bot("scripted")
#other_agent = make_bot("grodbot")

api.new_game(home_team_name="Human Team", away_team_name="Human Team",
             home_agent=make_bot("grodbot"), away_agent=other_agent)
server.start_server(debug=True, port=5005)
