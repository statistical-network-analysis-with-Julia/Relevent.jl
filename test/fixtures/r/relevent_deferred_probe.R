# Reproduce the deferred-effect discrepancies described in docs/src/guide/r_catalogue.md.
# Prints candidate log-rate matrices; this is evidence for a limitation, not a parity fixture.
# Reference inspected: R relevent 1.2.1; source git 7f7748ffa3b89bc8829f4e9159e76bbcb779611c.
suppressMessages(library(relevent))
ev <- rbind(c(1,1,2),c(2,1,2),c(3,1,3),c(4,3,1),c(5,2,3),c(6,4,2),c(7,2,1),c(8,1,4),c(9,1,2),c(10,3,2))
n <- 4L
nm <- c('NIDSnd','NIDRec','NODSnd','NODRec','NTDegSnd','NTDegRec','FrPSndSnd','FrRecSnd','RRecSnd','RSndSnd','OTPSnd','ITPSnd','OSPSnd','ISPSnd','FESnd','FERec','FEInt')
acl <- accum.interact(ev)
cumideg <- acl.deg(acl,n,'in'); cumodeg <- acl.deg(acl,n,'out')
rrl <- accum.rrl(ev); tri <- acl.tri(acl)
for (ef in nm) {
 model <- rem.dyad(NULL,n,effects=ef,coef.seed=rep(1,if(ef %in% c('FESnd','FERec','FEInt')) n-1 else 1),verbose=FALSE)
 for (iter in c(1L,2L,4L,10L)) {
 z <- rem.dyad.lambda(model$coef,iter,model$effects,n,nrow(ev),acl,cumideg,cumodeg,rrl,NULL,NULL,tri)
 cat(ef, 'iter',iter,'\n'); print(round(z,5))
 }
}
