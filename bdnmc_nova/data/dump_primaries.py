"""The generator's primaries in a NOvASoft art file, as CSV: one row per MCParticle of each simb::MCTruth.

Runs in the NOvASoft environment (the release's PyROOT and art dictionaries; reads the Events tree directly), e.g. at the end of stage 2 of
data/two_stage.sh (DUMP_PRIMARIES). The CSV is then compared with the TextFileGen input outside that environment
(jobs/check_textfilegen.py).

    python dump_primaries.py sim.root --out primaries.csv [--tag generator]

Columns: entry (art event order in the file), run, subrun, event, truth (index of the MCTruth in the event),
index (particle in the MCTruth), status, pdg, mother, x y z (cm) t (ns), px py pz E (GeV), mass (GeV).
"""
import argparse
import csv
import sys

COLUMNS = ["entry", "run", "subrun", "event", "truth", "index", "status", "pdg", "mother", "x", "y", "z", "t",
           "px", "py", "pz", "e", "mass"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("art_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="generator", help="input tag of the std::vector<simb::MCTruth> (module label)")
    a = ap.parse_args(argv)

    import ROOT  # from the NOvASoft release; the art dictionaries load through their rootmaps
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(a.art_file)
    tree = f.Get("Events")
    prefix = "simb::MCTruths_{}__".format(a.tag)
    names = [b.GetName() for b in tree.GetListOfBranches() if b.GetName().startswith(prefix)]
    if len(names) != 1:
        sys.exit("{}: {} branches named {}*: {}".format(a.art_file, len(names), prefix, names))
    branch = names[0]
    tree.SetBranchStatus("*", 0)
    for b in (branch, "EventAuxiliary"):
        tree.SetBranchStatus(b + "*", 1)
    rows = 0
    with open(a.out, "w") as out:
        w = csv.writer(out)
        w.writerow(COLUMNS)
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            aux = getattr(tree, "EventAuxiliary")
            truths = getattr(tree, branch).product()
            for it in range(truths.size()):
                mct = truths.at(it)
                for ip in range(mct.NParticles()):
                    p = mct.GetParticle(ip)
                    w.writerow([entry, aux.run(), aux.subRun(), aux.event(), it, ip, p.StatusCode(), p.PdgCode(),
                                p.Mother()] + [repr(float(v)) for v in (p.Vx(), p.Vy(), p.Vz(), p.T(), p.Px(), p.Py(),
                                                                       p.Pz(), p.E(), p.Mass())])
                    rows += 1
    entries = tree.GetEntries()
    sys.stdout.write("{} events, {} primaries -> {}\n".format(entries, rows, a.out))


if __name__ == "__main__":
    main()
