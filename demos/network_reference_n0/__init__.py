"""N0: reproduction of the Gutter Synthesis sound core and its 8-node network (no CA yet).

Modules
  gutter_node.py     scalar 1:1 port of the shipped gutterOsc.class (verification reference)
  gutter_network.py  vectorised 8-node network + the Max-side signal chain (numpy)
  source_manifest.py source commit / file hashes / extracted parameter table
  verify_node.py     port vs. original class executed on a JVM (needs --jdk and --source)
  render_n0.py       reproducible build of demos/results/network_reference_n0/
"""
