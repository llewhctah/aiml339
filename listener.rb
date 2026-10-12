# Sonic Pi listener for the live hand-detection system.
# Run this in Sonic Pi before starting camera.py.
# Plays a beep for each note number received
live_loop :listen do
  use_real_time
  note = sync "/osc*/note/play"
  synth :beep, note: note[0]
end