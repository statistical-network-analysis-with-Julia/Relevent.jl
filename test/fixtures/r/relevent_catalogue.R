# Regenerate from Relevent.jl: Rscript test/fixtures/r/relevent_catalogue.R > test/fixtures/relevent_catalogue.toml
# Each unit coefficient exposes the pre-event design statistic through R's
# rem.dyad.lambda (which returns LOG rates). No optimizer or RNG is involved.
# Repeated dyads and isolated actors distinguish event counts from binary ties;
# nonchronological actor appearances distinguish ID order from list positions.
# Unsupported FrPSndSnd/FrRecSnd and OSPSnd are deliberately excluded: see
# docs/src/guide/r_catalogue.md for the upstream source discrepancies.
suppressMessages(library(relevent))
n <- 5L
ev <- rbind(c(1,1,2),c(2,1,2),c(3,1,3),c(4,3,1),c(5,2,3),
            c(6,4,2),c(7,2,1),c(8,1,4),c(9,1,2),c(10,3,2),
            c(11,3,2),c(12,2,3),c(13,3,4),c(14,4,1))
m <- nrow(ev)
acl <- accum.interact(ev)
ideg <- acl.deg(acl,n,'in'); odeg <- acl.deg(acl,n,'out')
rrl <- accum.rrl(ev); tri <- acl.tri(acl)
num <- function(x) paste(sprintf('%.17g',x),collapse=', ')
cat('name = "relevent_catalogue"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\nrelevent_version = "%s"\n',getRversion(),packageVersion('relevent')))
cat('seed = 0 # deterministic; no random draws\n')
cat('script = "test/fixtures/r/relevent_catalogue.R"\n')
cat(sprintf('date = "%s"\n',Sys.Date()))
cat('dataset = "14 fixed directed events, 5 actors; actor 5 isolated; all candidate dyads before each event"\n')
cat('method = "unit-coefficient rem.dyad.lambda log-rate design, cumulative unit-event counts"\n\n')
cat('[tolerance]\n# Deterministic integer counts/ranks; only Float64 division roundoff.\ndefault = 1e-12\n\n[values]\n')
cat(sprintf('n_actors = %d\ninput_time = [%s]\ninput_sender = [%s]\ninput_receiver = [%s]\n',n,num(ev[,1]),num(ev[,2]),num(ev[,3])))
for (ef in c('NIDSnd','NIDRec','NODSnd','NODRec','NTDegSnd','NTDegRec',
             'RRecSnd','RSndSnd','OTPSnd','ITPSnd','ISPSnd','FESnd','FERec','FEInt','CovEvent')) {
  fixed <- ef %in% c('FESnd','FERec','FEInt')
  eventcov <- outer(seq_len(n),seq_len(n),function(i,j) (3*i-j)/7)
  covar <- if(ef == 'CovEvent') list(CovEvent=eventcov) else NULL
  for (col in seq_len(if(fixed) n-1L else 1L)) {
    pv <- rep(0,if(fixed) n-1L else 1L); pv[col] <- 1
    model <- rem.dyad(NULL,n,effects=ef,coef.seed=pv,covar=covar,verbose=FALSE)
    prepared <- if(ef == 'CovEvent') covarPrep(covar,n=n,m=m,effects=model$effects) else NULL
    values <- c()
    for (iter in seq_len(m)) {
      z <- rem.dyad.lambda(pv,iter,model$effects,n,m,acl,ideg,odeg,rrl,prepared,NULL,tri)
      for (s in seq_len(n)) for(r in seq_len(n)) if(s != r) values <- c(values,z[s,r])
    }
    key <- if(fixed) paste0(ef,'_',col+1L) else ef
    cat(sprintf('%s = [%s]\n',key,num(values)))
  }
}
