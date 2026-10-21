LoadPackage("grape");


# Returns the set of all nontrivial differences
DifferenceSet := function(set)
    local differences, x, y;
    differences := [];
    for x in set do
        for y in set do
            if x <> y then
                Add(differences, x * y^-1);
            fi;
        od;
    od;
    return Set(differences);
end;


# Next function assumes:
#   1. set is already admissible;
#   2. differences = DifferenceSet(set).
# It only checks the new differences introduced by adding a.

AdmissibleExtensionFast := function(set, differences, a)
    local x, ainv, d1, d2, newdifferences;
    if a in set then
        return false;
    fi;
    ainv := a^-1;
    newdifferences := [];
    for x in set do
        # New difference a * x^-1
        d1 := a * x^-1;
        if d1 in differences then
            return false;
        fi;
        if d1 in newdifferences then
            return false;
        fi;
        Add(newdifferences, d1);
        # New difference x * a^-1
        d2 := x * ainv;
        if d2 in differences then
            return false;
        fi;
        if d2 in newdifferences then
            return false;
        fi;
        Add(newdifferences, d2);
    od;
    return true;

end;

# Safe standalone version.
# This does not assume that set is already admissible.

AdmissibleExtension := function(G, set, a)
    local newset, differences;
    if a in set then
        return false;
    fi;
    newset := Concatenation(set, [a]);
    differences := DifferenceSet(newset);
    return Length(differences)
           = Length(newset) * (Length(newset)-1);
end;


# Constructs the bipartite graph with vertex set G x {1,2} and
# (g,1) and (h,2) connected iff g^-1*h in set.
# TODO: encode the Singer action in the returned graph

GraphDiffSet := function(G, set)
    local vertices, relation, action, gamma;
    vertices := Concatenation(
        List(G, g -> [g,1]),
        List(G, g -> [g,2])
    );
    relation := function(v, w)
        if v[2] = 1 and w[2] = 2 then
            return v[1]^-1 * w[1] in set;
        elif w[2] = 1 and v[2] = 2 then
            return w[1]^-1 * v[1] in set;
        else
            return false;
        fi;
    end;
    action := function(v, g)
        return v;
    end;
    gamma := Graph(
        Group(()),
        vertices,
        action,
        relation,
        true
    );
    return gamma;
end;


# Computes admissible sets of size q+1 containing the identity.
#
# Optimizations:
#   1. All size-2 starting sets are checked for admissibility.
#   2. For each partial set, its old differences are computed only once.
#   3. Each candidate extension checks only the newly introduced
#      differences.
#   4. Literal duplicates are removed after every extension step.
#   5. Partial sets are reduced modulo Aut(G) after every extension step.
SingerSets := function(G, q)
    local aut, classes, c, sets, size, newsets, set, a, differences, orbs,
    gammas, elements;
    ###
    aut := AutomorphismGroup(G);
    elements := Elements(G);
    classes := Orbits(aut, elements);
    sets := [];
    #########################################################################
    # Initial admissible sets of size 2.
    # This check is essential.
    # For example, if c has order 2, then
    #       c * 1^-1 = 1 * c^-1,
    # so [1,c] is not admissible.
    #########################################################################
    for c in classes do
        c := c[1];
        if c <> One(G) then
            if AdmissibleExtension(G,[One(G)],c) then
                Add(sets,[One(G), c]);
            fi;
        fi;
    od;
    size := 2;
    repeat
        Print(
            "Extending Singer sets from size ",
            size,
            " to size ",
            size+1,
            ". Current number: ",
            Length(sets),
            "\n"
        );
        newsets := [];
        #####################################################################
        # Extend every orbit representative.
        #####################################################################
        for set in sets do
            differences := DifferenceSet(set);
            #################################################################
            # Safety check:
            # Every partial set reaching this point must already be
            # admissible.
            #################################################################
            if Length(differences) <>
               Length(set) * (Length(set)-1) then
                Error("Non-admissible partial Singer set encountered.");
            fi;
            for a in elements do
                if AdmissibleExtensionFast(set,differences,a) then
                    Add(newsets,Concatenation(set, [a]));
                fi;
            od;
        od;
        Print(
            "Before removing duplicates: ",
            Length(newsets),
            "\n"
        );
        #####################################################################
        # Remove exact duplicates.
        #####################################################################
        Perform(newsets, Sort);
        newsets := SSortedList(newsets);
        Print(
            "After removing literal duplicates: ",
            Length(newsets),
            "\n"
        );
        #####################################################################
        # Reduce modulo Aut(G).
        #
        # This is safe at every level:
        # if two partial sets are automorphic, extending one by all
        # elements of G represents all extensions of the other.
        #####################################################################
        if Length(newsets) > 0 then
            orbs := Orbits(aut,newsets,OnSets);
            sets := List(orbs,o -> o[1]);
        else
            sets := [];
        fi;
        Print(
            "After quotienting by automorphisms: ",
            Length(sets),
            "\n\n"
        );
        size := size + 1;
    until size = q+1 or sets = [];
    #########################################################################
    # Keep only sets that generate G.
    #
    # Since every set contains One(G), the corresponding bipartite graph
    # GraphDiffSet(G, set) is connected iff <set> = G.
    #########################################################################
    sets := Filtered(sets,set -> Group(set) = G);
    Print(
        "Found ",
        Length(sets),
        " connected Singer sets up to automorphisms.\n"
    );
    #########################################################################
    # Final algebraic safety check. TURNED OFF
    ########################################################################
    # for set in sets do
    #     differences := DifferenceSet(set);
    #     if Length(differences) <>
    #        Length(set) * (Length(set)-1) then
    #         Error(
    #             "Final Singer set failed the difference test."
    #         );
    #     fi;
    # od;
    #########################################################################
    # Final graph-theoretic plausibility check. TURNED OFF
    #########################################################################
    # gammas := List(sets,s -> GraphDiffSet(G, s));
    # if ForAny(gammas, g -> Girth(g) < 6) then
    #     Error("A calculated Singer set produced a graph of girth < 6.");
    # fi;
    return sets;
end;


SingerActions := function(G, q)
    local sets, graphs;
    Print("Computing Singer sets.\n");
    Print("Group order: ", Size(G), "\n");
    Print("Group description: ", StructureDescription(G), "\n");
    Print("Graph degree: ", q+1, "\n");
    Print("Number of graph vertices: ", 2*Size(G), "\n");
    Print("Number of graph edges: ", Size(G)*(q+1), "\n");
    sets := SingerSets(G, q);
    graphs := List(sets,set -> GraphDiffSet(G, set));
    Print("Constructed ",Length(graphs)," graphs.\n");
    Print("Now computing a representative system\n");
    graphs := GraphIsomorphismClassRepresentatives(graphs);
    Print("We now have ", Size(graphs), " graphs\n\n");
    return graphs;
end;

SingerActionsAllGroupsOrder := function(n, q)
    local groups, k, allgraphs, i, G, graphs;
    groups := AllSmallGroups(n);
    k := Length(groups);
    allgraphs := [];
    for i in [1..k] do
        Print("Now checking group ", i, "/", k,"\n");
        G := groups[i];
        graphs := SingerActions(G, q);
        Print("Found ",Length(graphs)," graphs for group ",i,".\n");
        Append(allgraphs,graphs);
        Print("Now computing a representative system for all graphs we found so far\n");
        allgraphs := GraphIsomorphismClassRepresentatives(allgraphs);
        Print("We now have ", Size(allgraphs), " graphs\n\n");
    od;
    return allgraphs;
end;

SingerActionsAllGroupsOrderNoAbelian := function(n, q)
    local groups, k, allgraphs, i, G, graphs;
    groups := AllSmallGroups(n);
    k := Length(groups);
    allgraphs := [];
    for i in [1..k] do
        Print("Now checking group ", i, "/", k,"\n");
        G := groups[i];
        if IsAbelian(G) then
            Print("Group is Abelian and ignored. \n\n");
        else   
            graphs := SingerActions(G, q);
            Print("Found ",Length(graphs)," graphs for group ",i,".\n");
            Append(allgraphs,graphs);
            Print("Now computing a representative system for all graphs we found so far\n");
            allgraphs := GraphIsomorphismClassRepresentatives(allgraphs);
            Print("We now have ", Size(allgraphs), " graphs\n\n");
        fi;
    od;
    return allgraphs;
end;

###############

SaveGraphs := function(graphs, filename)
    local g, n, edges, e;
    PrintTo(filename, "");
    for g in graphs do
        n := OrderGraph(g);
        edges := UndirectedEdges(g);
        AppendTo(filename, n, " ", Length(edges), "\n");
        for e in edges do
            AppendTo(filename, e[1] - 1, " ", e[2] - 1, "\n");
        od;
    od;
end;