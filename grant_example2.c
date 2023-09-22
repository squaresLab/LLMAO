        /* (A I PeriodicType 100)
           (A D A3 (6.2380000000000004 0 0))
           (A D B3 (0 6.9909999999999997 0))
           (A D C3 (0 0 6.9960000000000004))
           (A C SpaceGroup "63 5")
        */
        if (strstr(buffer, "PeriodicType") != nullptr) {
          ifs.getline(buffer,BUFF_SIZE); // next line should be translation vector
          tokenize(vs,buffer);
            while (vs.size() == 6) {
              x = atof((char*)vs[3].erase(0,1).c_str());
              y = atof((char*)vs[4].c_str());
              z = atof((char*)vs[5].c_str());
​
              translationVectors[numTranslationVectors++].Set(x, y, z);
              if (!ifs.getline(buffer,BUFF_SIZE))
                break;
              tokenize(vs,buffer);
            }
        }
​
        if (strstr(buffer, "SpaceGroup") != nullptr) {
          tokenize(vs, buffer);
          if (vs.size() != 5)
            continue; // invalid space group
          setSpaceGroup = true;
          sg = SpaceGroup::GetSpaceGroup(vs[4]); // remove the initial " character
        }
​
        // atom information
        if (atomRecord) {
          if (strstr(buffer, "ACL") != nullptr) {
            tokenize(vs, buffer);
            // size should be 5 -- need a test here
            if (vs.size() != 5) return false; // timvdm 18/06/2008
            vs[3].erase(0,1); // "6 => remove the first " character
            unsigned int atomicNum = atoi(vs[3].c_str());
            if (atomicNum == 0)
              atomicNum = 1; // hydrogen ?
​
            // valid element, so create the atom
            atom = mol.NewAtom();
            atom->SetAtomicNum(atomicNum);
            continue;
          }
          else if (strstr(buffer, "XYZ") != nullptr) {
            tokenize(vs, buffer);
            // size should be 6 -- need a test here
            if (vs.size() != 6) return false; // timvdm 18/06/2008
            vs[3].erase(0,1); // remove ( character
            vs[5].erase(vs[5].length()-2, 2); // remove trailing )) characters
            atom->SetVector(atof(vs[3].c_str()),
                            atof(vs[4].c_str()),
                            atof(vs[5].c_str()));
            continue;
          }
        } // end of atom records
​
        // bond information
        if (bondRecord) {
          if (strstr(buffer, "Atom1") != nullptr) {
            tokenize(vs, buffer);
            if (vs.size() < 4) return false; // timvdm 18/06/2008
            vs[3].erase(vs[3].length()-1,1);
            startBondAtom = atoi(vs[3].c_str());
            continue;
          }
          else if (strstr(buffer, "Atom2") != nullptr) {
            tokenize(vs, buffer);
            if (vs.size() < 4) return false; // timvdm 18/06/2008
            vs[3].erase(vs[3].length()-1,1);
            endBondAtom = atoi(vs[3].c_str());
            continue;
          }
          else if (strstr(buffer, "Type") != nullptr) {
            tokenize(vs, buffer);
            if (vs.size() < 4) return false; // timvdm 18/06/2008
            vs[3].erase(vs[3].length()-1,1);
            bondOrder = atoi(vs[3].c_str());
            if (bondOrder == 4) // triple bond?
              bondOrder = 3;
            else if (bondOrder == 8) // aromatic?
              bondOrder = 5;
            else if (bondOrder != 2) // 1 OK, 2 OK, others unknown
              bondOrder = 1;
            continue;
          }
        }
​
        // ending a "tag" -- a lone ")" on a line
        if (strstr(buffer, ")") != nullptr && strstr(buffer, "(") == nullptr) {
          openParens--;
          if (atomRecord) {
            atomRecord = false;
          }
          if (bondRecord) {
            // Bond records appear to be questionable
            mol.AddBond(startBondAtom - 1, endBondAtom - 1, bondOrder);
            bondRecord = false;
          }
​
          if (openParens == 0) {
            ifs.getline(buffer, BUFF_SIZE);
            break; // closed this molecule
          }
        }
      }