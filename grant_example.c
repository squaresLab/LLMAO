 bool MSIFormat::ReadMolecule(OBBase* pOb, OBConversion* pConv)
  {
    OBMol* pmol = pOb->CastAndClear<OBMol>();
    if (pmol == nullptr)
      return false;
​
    //Define some references so we can use the old parameter names
    istream &ifs = *pConv->GetInStream();
    OBMol &mol = *pmol;
    const char* title = pConv->GetTitle();
    char buffer[BUFF_SIZE];
​
    stringstream errorMsg;
​
    if (!ifs)
      return false; // we're attempting to read past the end of the file
​
    if (!ifs.getline(buffer,BUFF_SIZE))
      {
        obErrorLog.ThrowError(__FUNCTION__,
                              "Problems reading an MSI file: Cannot read the first line.", obWarning);
        return(false);
      }
​
    if (!EQn(buffer, "# MSI CERIUS2 DataModel File", 28))
      {
        obErrorLog.ThrowError(__FUNCTION__,
                              "Problems reading an MSI file: The first line must contain the MSI header.", obWarning);
        return(false);
      }
​
    // "records" start with
    // (1 Model
    // ....
    //   and end with
    // ....
    // )
    unsigned int openParens = 0; // the count of "open parentheses" tags
    unsigned int startBondAtom, endBondAtom, bondOrder;
    bool atomRecord = false;
    bool bondRecord = false;
    OBAtom *atom;
//    OBBond *bond;
    vector<string> vs;
    const SpaceGroup *sg;
    bool setSpaceGroup = false;
    double x,y,z;
    vector3 translationVectors[3];
    int numTranslationVectors = 0;
​
    mol.BeginModify();
    while (ifs.getline(buffer,BUFF_SIZE))
      {
        // model record
        if (strstr(buffer, "Model") != nullptr) {
          openParens++;
          continue;
        }
​
        // atom record
        if (!bondRecord && strstr(buffer, "Atom") != nullptr) {
          atomRecord = true;
          openParens++;
          continue;
        }
​
        if (strstr(buffer, "Bond") != nullptr) {
          bondRecord = true;
          startBondAtom = endBondAtom = 0;
          bondOrder = 1;
          openParens++;
          continue;
        }
​
